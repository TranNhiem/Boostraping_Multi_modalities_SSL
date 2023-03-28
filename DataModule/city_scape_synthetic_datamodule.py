'''
TranNhiem 2022/03 
This file is used to create datamodule for cityscape dataset

The Generated Pipeline is as follow:

    Step 1. Load data from folder
    Step 2. Resize image to 256x512 and Transform Image to Tensor
    Step 3 (3. BLIP2 Model generate Caption for the Image, Visual Dialog from ChatGPT+BLIP2) 
    Step 4 . Generate the image condition (Edges, Segmentation, Depth+Normal map, others)
        + Condition Images generate from ControlNet pretrained condition model
        + Condition Images generate from Prismer (A Vision-Language Model)
    Step 5. Generate Image (Text, Initial Image, Condition Image) 
    
'''

##**************************************************************
## Step 1 && 2 : Dataloader to Load CityScape Dataset
##**************************************************************

##1 Load image from Folder 
##2 Resize image to 2x Down scale (1024x512) 4x Down scale (512, 256) 

import os
import torch
from torchvision import transforms
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CityscapesSegmentation(Dataset):
    '''
    Data Structure of CityScape Dataset
        cityscapes
    ├── gtFine # Segmentation Label
    │   ├── test (19 City Folders)
    │   ├── train (19 City Folders)
    │   └── val (19 City Folders)
    └── leftImg8bit
        ├── test (19 City Folders)
        ├── train (19 City Folders)
        └── val (19 City Folders)

    '''

    def __init__(self, root_dir, split, transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        self.image_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)

        self.images = []
        for city_dir in os.listdir(self.image_dir):
            city_image_dir = os.path.join(self.image_dir, city_dir)
            for image_name in os.listdir(city_image_dir):
                image_path = os.path.join(city_image_dir, image_name)
                label_name = '{}_gtFine_color.png'.format(image_name.split('_leftImg8bit')[0])
                label_path = os.path.join(self.label_dir, city_dir, label_name)
                assert os.path.isfile(image_path), f"{image_path} does not exist"
                assert os.path.isfile(label_path), f"{label_path} does not exist"
                self.images.append((image_path, label_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        
        if self.transforms is not None:
            image = self.transforms(image)
            ## Resize lable image to match the size of the image
            #label = transforms.Resize(image.shape[1:], Image.NEAREST)(label)
            img_size=image.size
            label = transforms.Resize((img_size[1],img_size[0]), Image.NEAREST)(label)
            #breakpoint()
        return image, label


data_dir = '/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/cityscape_synthetic/'
batch_size = 256
split = 'train'
re_size = (256, 512)

transform = transforms.Compose([
    transforms.Resize((re_size)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ## convert back to PIL image
    transforms.ToPILImage()
])
dataset = CityscapesSegmentation(root_dir=data_dir, split='train', transforms=transform)

## Testing iterate through the dataset to get images and labels
# for i, (image, label) in enumerate(dataset):
#     # do something with the image and label, for example print their shapes
#     print(f"Image shape: {image.size}, Label shape: {label.size}")
#     label.save('test_segment.png')
#     image.save('test_image.png')
#     break




##**************************************************************
## Step 3 Generate Caption for the Image USING BLIP2 Model
##**************************************************************
'''
## Existing Pretraining BLIP Model 

1.. Salesforce/blip2-flan-t5-xxl
2.. Salesforce/blip2-flan-t5-xl
2.. Salesforce/blip2-opt-6.7b 
3.. Salesforce/blip2-opt-2.7b

Optimized Fine-tuned COCO Captioning
Salesforce/blip2-opt-6.7b-coco 
Salesforce/blip2-opt-2.7b-coco
Salesforce/blip2-flan-t5-xl-coco 

'''

### Using BLIP 2 
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration

weight_path= "/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weight/BLIP2/"
## check the weight path if not create the path
if not os.path.exists(weight_path):
    os.makedirs(weight_path)

# device = "cuda" if torch.cuda.is_available() else "cpu"

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco", cache_dir=weight_path )

# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b-coco",cache_dir=weight_path,  torch_dtype=torch.float16
# )
# model.to(device)

# for i, (image, label) in enumerate(dataset):
#     # do something with the image and label, for example print their shapes
#     print(f"Image shape: {image.size}, Label shape: {label.size}")
#     # label.save('test_segment.png')
#     # image.save('test_image.png')
   
# # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# # image = Image.open(requests.get(url, stream=True).raw)

#     inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
#     generated_ids = model.generate(**inputs,  penalty_alpha=0.6, top_k=4)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     print(generated_text)
    
#     break

##********************************************************************************************
## Step 3 Generate Caption ChatGPT & BLIP2 (Task Instructed Instructed)
##********************************************************************************************

import sys
sys.path.append('/home/rick/DataEngine_Pro/Boostraping_Multi_modalities_SSL/DataModule/ChatCaptioner')
import yaml
import torch

from chatcaptioner.chat import set_openai_key, caption_images, get_instructions
from chatcaptioner.blip2 import Blip2
from chatcaptioner.utils import RandomSampledDataset, plot_img, print_info

openai_key = os.environ["OPENAI_API_KEY"]
set_openai_key(openai_key)

blip2s = {
    #'FlanT5 XXL': Blip2('FlanT5 XXL', device_id=0, bit8=True), # load BLIP-2 FlanT5 XXL to GPU0. Too large, need 8 bit. About 20GB GPU Memory
     'OPT2.7B COCO': Blip2('OPT2.7B COCO', device_id=1, bit8=False), # load BLIP-2 OPT2.7B COCO to GPU1. About 10GB GPU Memory
    # 'OPT6.7B COCO': Blip2('OPT6.7B COCO', device_id=2, bit8=True), # load BLIP-2 OPT6.7B COCO to GPU2. Too large, need 8 bit.
}
blip2s_q = {}

## ------------ Setting the Parameters -----------------

# set the dataset to test
dataset_name = 'cityscape_train'  # current options: 'artemis', 'cc_val', 'coco_val'
# set the number of chat rounds between GPT3 and BLIP-2
n_rounds = 8
# set the number of visible chat rounds to BLIP-2. <0 means all the chat histories are visible.
n_blip2_context = 1
# if print the chat out in the testing
print_chat = True
# set the question model
question_model_tag = 'gpt-3.5-turbo'

## ------------ Loading the Dataset -----------------

# preparing the folder to save results
SAVE_PATH = '/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/cityscape_synthetic/{}/{}'.format(question_model_tag, dataset_name)
if not os.path.exists(SAVE_PATH):
    os.makedirs(os.path.join(SAVE_PATH, 'caption_result'))
with open(os.path.join(SAVE_PATH, 'instruction.yaml'), 'w') as f:
    yaml.dump(get_instructions(), f)

if question_model_tag in blip2s_q:
    question_model = blip2s_q[question_model_tag]
else:
    question_model = question_model_tag

# ------------ Testing Generate Question-----------------
for i, (image, label) in enumerate(dataset):
    sample_img_ids=i
    image.save(f'test_image_{i}.png')
    caption_images(blip2s, 
                image, 
                sample_img_ids, 
                save_path=SAVE_PATH, 
                n_rounds=n_rounds, 
                n_blip2_context=n_blip2_context,
                model=question_model,
                print_mode='chat')
    
    if i==4:
        break 


##********************************************************************************************
## Step 4 Generate Conditional Generate Image Condition 
##********************************************************************************************
# Load 