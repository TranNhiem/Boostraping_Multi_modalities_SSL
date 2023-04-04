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

    def __init__(self, root_dir, split, transforms=None, version='gtFine'):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms

        if version=='gtFine':
            self.image_dir = os.path.join(root_dir, 'leftImg8bit', split)
            self.label_dir = os.path.join(root_dir, 'gtFine', split)
        else:
        
            self.image_dir = os.path.join(root_dir, 'leftImg8bit', split)
            self.label_dir = os.path.join(root_dir, 'gtCoarse', split)

        self.images = []
        for city_dir in os.listdir(self.image_dir):
            city_image_dir = os.path.join(self.image_dir, city_dir)
            for image_name in os.listdir(city_image_dir):
                image_path = os.path.join(city_image_dir, image_name)
                if version=='gtFine':
                    label_name = '{}_gtFine_color.png'.format(image_name.split('_leftImg8bit')[0])
                else:
                    label_name = '{}_gtCoarse_color.png'.format(image_name.split('_leftImg8bit')[0])
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
dataset = CityscapesSegmentation(root_dir=data_dir, split='train_extra', transforms=transform, version="gtCoarse")

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
# if not os.path.exists(weight_path):
#     os.makedirs(weight_path)

#device = "cuda" #if torch.cuda.is_available() else "cpu"

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco", cache_dir=weight_path )

# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b-coco",cache_dir=weight_path,  torch_dtype=torch.float16
# )
# model.to(device)

# for i, (image, label) in enumerate(dataset):
#     # do something with the image and label, for example print their shapes
#     print(f"Image shape: {image.size}, Label shape: {label.size}") #2048,1024 --> 512, 256
#     # label.save('./temp_imgs/test_segment.png')
#     # image.save('./temp_imgs/test_image.png')
   
# # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# # image = Image.open(requests.get(url, stream=True).raw)

#     inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
#     #generated_ids = model.generate(**inputs,  penalty_alpha=0.6, top_k=4)
#    ## Beam search, Neucleus, Contrastive 
#     generated_ids = model.generate(**inputs,   num_beams=5, early_stopping=True)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     print(generated_text)
#     if i==5: 
#         break

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
# openai.api_type = "azure"
# openai.api_base = "https://sslgroupservice.openai.azure.com/"
# openai.api_version = "2023-03-15-preview"
openai_key = os.environ["OPENAI_API_KEY"]
#CUDA_VISIBLE_DEVICES="1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
## export OPENAI_API_KEY=0aee54a3f2df4c55aea57bf3cf2e99a6
set_openai_key(openai_key)
## Adding These Line of code in Chat.py if using Azure OpenAI
'''
    VALID_CHATGPT_MODELS = ['gpt-3.5-turbo', "gpt-35-turbo"]## lINE 54
    from Line 74-> 77
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview" 
    openai.api_base = "https://sslgroupservice.openai.azure.com/"#os.getenv("OPENAI_API_BASE")  # Your Azure OpenAI resource's endpoint value.
    openai.api_key = os.getenv("OPENAI_API_KEY")

    question_model_tag ="gpt-35-turbo"

''' 
    
device = "cuda" if torch.cuda.is_available() else "cpu"

blip2s = {
    #'FlanT5 XXL': Blip2('FlanT5 XXL', device_id=0, bit8=True), # load BLIP-2 FlanT5 XXL to GPU0. Too large, need 8 bit. About 20GB GPU Memory
     'OPT2.7B COCO': Blip2('OPT2.7B COCO', device_id=2, bit8=False), # load BLIP-2 OPT2.7B COCO to GPU1. About 10GB GPU Memory
    # 'OPT6.7B COCO': Blip2('OPT6.7B COCO', device_id=2, bit8=True), # load BLIP-2 OPT6.7B COCO to GPU2. Too large, need 8 bit.
}
blip2s_q = {}

## ------------ Setting the Parameters -----------------

# set the dataset to test
dataset_name = 'cityscape_train_coarse'  # current options: 'artemis', 'cc_val', 'coco_val'
# set the number of chat rounds between GPT3 and BLIP-2
n_rounds = 8
# set the number of visible chat rounds to BLIP-2. <0 means all the chat histories are visible.
n_blip2_context = 1
# if print the chat out in the testing
print_chat = True
# set the question model
question_model_tag ="gpt-35-turbo"## for OPENAI "gpt-3.5-turbo"

# ------------ Loading the Dataset -----------------
## preparing the folder to save results
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
#for i, (image, label) in enumerate(dataset):
# for i, (image, label) in enumerate(dataset):
#     sample_img_ids=i
#     #image.save(f'./temp_imgs/test_image_{i}.png')
#     caption_images(blip2s, 
#                 image, 
#                 sample_img_ids, 
#                 save_path=SAVE_PATH, 
#                 n_rounds=n_rounds, 
#                 n_blip2_context=n_blip2_context,
#                 model=question_model,
#                 print_mode='chat')
    
#     if i==5000:
#         break 

for i, (image, label) in enumerate(dataset[5000:]):
    sample_img_ids = i + 5000
    image.save(f'./temp_imgs/test_image_{sample_img_ids}.png')
    caption_images(blip2s, 
                image, 
                sample_img_ids, 
                save_path=SAVE_PATH, 
                n_rounds=n_rounds, 
                n_blip2_context=n_blip2_context,
                model=question_model,
                print_mode='chat')

    if i==10000:
        break 

##********************************************************************************************
## Step 4 Generate Conditional Generate Image Condition 
##********************************************************************************************
'''
0. Testing Code with Image Variant usign SD 
    + UnCLIP Image Interpolation (UnCLIP Image Interpolation Pipeline)
    + Image Variant with CLIP  Image Embedding
0.1 Using Pain by Sample to Generate Image Variant 
1. Consider CLIP Segmentation --> Semantic Segmentation via Text Description
2. Grismer multi modalities Condition --> 
3. Using ControlNet Condition 
'''

##********************************* Image Generation with SD image Invariant*****************************************
##-------------------------------------------
## 1 CLIP Image Embedding --> Image Variant
##-------------------------------------------

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImageVariationPipeline, StableDiffusionDepth2ImgPipeline


####-------------------------------------------
## Stable Diffusion Model with Image Variant Model Version 1 CLIP Image Embedding
####-------------------------------------------

weight_path= "/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weight/StableDiffusion/"


# sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
#             "lambdalabs/sd-image-variations-diffusers",
#             revision="v2.0",
#         torch_dtype=torch.float16,
#         use_auth_token=True,
#         cache_dir= weight_path,      
#     )
# def dummy(images, **kwargs): return images, False
# sd_pipe.safety_checker = dummy


## Using DPPM++ 
# sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)

# generative_model=sd_pipe.to("cuda")

# num_samples =3
# num_steps=30
# guidance_scale=8.5
# for i, (image, label) in enumerate(dataset):
#     init_img=image
#     image =generative_model(num_samples*[init_img],num_inference_steps=num_steps, guidance_scale=guidance_scale).images
#     for idx, im in enumerate(image):
#         output_path="./temp_imgs/" + str("invar_")+ str(i)+ str(idx) + ".jpg"
#         #output_path=os.path.join(output_path,str(idx) + ".jpg")
#         im.save(output_path)    
#     if i==3: 
#         break 


from diffusers import UniPCMultistepScheduler

# sd_pipe.scheduler = UniPCMultistepScheduler.from_config(sd_pipe.scheduler.config)
# #sd_pipe.enable_model_cpu_offload()
# sd_pipe.enable_xformers_memory_efficient_attention()

# generative_model=sd_pipe.to("cuda")
# num_samples =3
# num_steps=30
# guidance_scale=8.5

# for i, (image, label) in enumerate(dataset):
#     init_img=image
#     image =generative_model(num_samples*[init_img],num_inference_steps=num_steps, guidance_scale=guidance_scale).images
#     for idx, im in enumerate(image):
#         output_path="./temp_imgs/" + str("invar_UniPCM")+ str(i)+ str(idx) + ".jpg"
#         #output_path=os.path.join(output_path,str(idx) + ".jpg")
#         im.save(output_path)
    
#     if i==3: 
#         break
# 

####-------------------------------------------
## Stable Diffusion Model with Image Variant Model Version 2, Using CLIP Image Embedding
####-------------------------------------------
from diffusers import StableUnCLIPImg2ImgPipeline
# pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"

# )
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# #sd_pipe.enable_model_cpu_offload()
# pipe.enable_xformers_memory_efficient_attention()
# pipe = pipe.to("cuda")

# num_steps=20
# guidance_scale=8.5
# for i, (image, label) in enumerate(dataset):
#     init_img=image
#     image =pipe(image=init_img, prompt="", num_images_per_prompt=3, num_inference_steps=num_steps, guidance_scale=guidance_scale).images
#     for idx, im in enumerate(image):
#         output_path="./temp_imgs/" + str("invar_SD_V2")+ str(i)+ str(idx) + ".jpg"
#         #output_path=os.path.join(output_path,str(idx) + ".jpg")
#         im.save(output_path)
    
#     if i==3: 
#         break
# # 


##-------------------------------------------
## 2 UnCLIP Image Interpolation --> Image Variant
##-------------------------------------------

from diffusers import DiffusionPipeline

# device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
# dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
# pipe = DiffusionPipeline.from_pretrained(
#     "kakaobrain/karlo-v1-alpha-image-variations",
#     torch_dtype=dtype,
#     custom_pipeline="unclip_image_interpolation", 
#     cache_dir= weight_path,
# )
#pipe.enable_sequential_cpu_offload()
#pipe.enable_xformers_memory_efficient_attention()

# pipe=pipe.to(device)
# images=[]
# generator = torch.Generator(device=device).manual_seed(42)
# for i, (image, label) in enumerate(dataset):
#     #images.append(image)
#     #image.save(f'./temp_imgs/original_img_street%s_{i}.jpg')
#     if i>= 1: 
#         images.append(image)
        
#         if i%2 ==0:
#             for id, image in enumerate(images):
#                 image.save(f'./temp_imgs/original_img_street%s{i}_{id}.jpg')
#             output = pipe(image = images ,steps= 3,   decoder_num_inference_steps=30,super_res_num_inference_steps =10,  generator = generator)
#             ## 
#             for j,image in enumerate(output.images):
#                 image.save(f'./temp_imgs/interpol_street%s{i}_{j}.jpg')
#             images=[]
        
#     if i ==6: 
#         break 


### I

### ControlnET
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import controlnet_hinter
from diffusers import UniPCMultistepScheduler

# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd",  cache_dir=weight_path, torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", cache_dir=weight_path, controlnet=controlnet, torch_dtype=torch.float16
# )
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()
# pipe.enable_xformers_memory_efficient_attention()
# # output = pipe(
# pipe=pipe.to("cuda")
# device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

# generator = torch.Generator(device=device).manual_seed(42)
# prompt=""
# for i, (image, label) in enumerate(dataset):
#     mlsd_img= controlnet_hinter.hint_hough(image)
#     mlsd_img.save(f"./mlsd_img{i}.png")

#     output = pipe(
#     prompt,
#     mlsd_img,
#     negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality" ,
#     num_inference_steps=20,
#     generator=generator,
# )
#     image=output[0][0]
#     image.save(f"./temp_imgs/hough_img_output_{i}.png")
#     if i==3: 
#         break


# #### ControlNet Scribble Edge Conditioning 
# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble",  cache_dir=weight_path, torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", cache_dir=weight_path, controlnet=controlnet, torch_dtype=torch.float16
# )
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# #pipe.enable_model_cpu_offload()
# pipe.enable_xformers_memory_efficient_attention()
# # output = pipe(
# pipe=pipe.to("cuda")
# generator = torch.Generator(device=device).manual_seed(42)
# prompt=""
# for i, (image, label) in enumerate(dataset):
#     mlsd_img= controlnet_hinter.hint_scribble(image)
#     mlsd_img.save(f"./scribble_img{i}.png")

#     output = pipe(
#     prompt,
#     mlsd_img,
#     negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality" ,
#     num_inference_steps=20,
#     generator=generator,
# )
#     image=output[0][0]
#     image.save(f"./temps_imgs/scribble_img_output_{i}.png")



# #### ControlNet Scribble Segmmentation Conditioning 
# controlnet_1 = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",  cache_dir=weight_path, torch_dtype=torch.float16)
# controlnet_2 = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth",  cache_dir=weight_path, torch_dtype=torch.float16)
# controlnet= [controlnet_2, controlnet_1]
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", cache_dir=weight_path, controlnet=controlnet, torch_dtype=torch.float16
# )

# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()
# pipe.enable_xformers_memory_efficient_attention()
# # output = pipe(
# pipe=pipe.to("cuda")
# device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
# generator = torch.Generator(device=device).manual_seed(42)

# prompt="The image is of a city street with buildings and white cars parked on the street. The weather is cloudy and the predominant color of the buildings is gray. The shapes of the buildings are not specified."

# for i, (image, label) in enumerate(dataset):
#     segmentation_img= controlnet_hinter.hint_canny(image)
#     segmentation_img.save(f"./temp_imgs/canny_img_output_{i}.png")
#     depth_img= controlnet_hinter.hint_depth(image)
#     depth_img.save(f"./temp_imgs/depth_img_output_{i}.png")
#     images=[segmentation_img, depth_img]
#     output = pipe(
#     prompt,
#     images,
#     negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality" ,
#     num_inference_steps=20,
#     generator=generator,
# )
#     image=output[0][0]
#     image.save(f"./temp_imgs/Scribbel_depth_img_output_{i}.png")

#     if i==3: 
#         break
    