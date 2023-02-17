import os
import re
import json
import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
#from stable_diffusion_model import StableDiffusionPipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImageVariationPipeline, StableDiffusionDepth2ImgPipeline
from torchvision import transforms
from PIL import Image
import PIL
import sys
from pathlib import Path
from tqdm import tqdm

#CUDA_VISIBLE_DEVICES="1"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# from min_dalle import MinDalle
#CUDA_VISIBLE_DEVICES=2,3 python xxx.py
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f"Using available: {device}")

# Preprocessing the caption with some special characters

def pre_caption(caption, max_words=50):
    caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
    caption = re.sub(r"\s{2,}", ' ', caption)
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # Truncate caption
    caption_words = caption.split(' ')
    caption = caption.strip(' ')

    # Truncate cpation
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption_words = caption_words[:max_words]
        caption = ' '.join(caption_words)

    return caption


def dummy(images, **kwargs): return images, False

## ----------------------- 1-- TEXT to IMAGE ------------------------
## Generated data given by the text description
class Old_version(Dataset):

    def __init__(self, image_root, ann_root, max_words=200, prompt='4k , highly detailed', generate_mode="repeat", 
                         guidance_scale=7.5,  num_inference_steps=70, seed=123245):
        '''
        image_root (string): Root directory for storing the generated images (ex: /data/coco_synthetic/)
        anno_root(string): directory for storing the human caption file from COCO Caption dataset
        '''
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'

        Path(image_root + "val2014/").mkdir(parents=True, exist_ok=True)
        Path(image_root + "train2014/").mkdir(parents=True, exist_ok=True)

        Path(ann_root).mkdir(parents=True, exist_ok=True)

        # os.makedirs(image_root+ "val2014/", exist_ok=True)
        # os.makedirs(ann_root, exist_ok=True)

        download_url(url, ann_root)
        self.annotation = json.load(
            open(os.path.join(ann_root, filename), 'r'))

        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generate_mode= generate_mode
        # random.randint(0, 100000) #random.randint(0,10000) # change the seed to get different results
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        self.append_id = ["test"]
        self.repeat_name=["test"]
        self.append_id_repeat=["test"]
        
        
        ### Stable Diffusion 1.4
        store_path="/data1/pretrained_weight/StableDiffusion/"
        # self.model = StableDiffusionPipeline.from_pretrained(
        #     "CompVis/stable-diffusion-v1-4", revision="fp16",
        #     torch_dtype=torch.float32,
        #     use_auth_token=True,
        #     cache_dir= store_path, 
        # ).to("cuda")

        ### Stable Diffusion 2.1
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=True,
            cache_dir= store_path, 
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.model=pipe.to("cuda")
        

        
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        ann = self.annotation[idx]

        caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        image_id = ann['image_id']
        image_name = ann['image']  # Saved image's name
        path=os.path.join(self.image_root, image_name)
        if self.generate_mode=="repeat":
            if image_id == self.append_id[-1]:
                #print("Using mode Image to generate image")
                init_image = Image.open(os.path.join(self.image_root, image_name))
                with torch.autocast('cuda'):
                    generate_image = self.model(
                        prompt=[caption],
                        mode="image",
                        height=512,
                        width=512,
                        num_inference_steps=50,
                        guidance_scale=self.guidance_scale,
                        init_image=init_image,
                        generator=self.generator,
                        strength=0.8,
                        return_intermediates=False,
                    ).images[0]

            else:  # Case not repeat image
                #print("Using mode Prompt to generate image")
                        with torch.autocast('cuda'):
                            generate_image = self.model(
                                prompt=[caption],
                                mode="prompt",
                                height=512,
                                width=512,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                init_image=None,
                                generator=self.generator,
                                strength=0.8,
                                return_intermediates=False,
                            ).images[0]
        else: 
            ## inCase the image name repeat 
            if image_id == self.append_id[-1]:
                # The first image repeat is creat
                image_name_= self.repeat_name[-1][:-5] + "1" +".jpg"
                path=os.path.join(self.image_root, image_name_)

                #checking image is exist or not
                if os.path.isfile(path) is True or os.path.exists(path) is True:
                    print("Next repeat image is append")
                    image_name= self.repeat_name[-1][:-4] + "1" + ".jpg"
                    image_id=self.append_id_repeat[-1] + "1"
                    path=os.path.join(self.image_root, image_name)
                    self.append_id_repeat.append(image_id)
                    self.repeat_name.append(image_name)

                else:
                    print("first repeat image is created")
                    image_id= image_id +"1"
                    image_name= image_name[:-4] + "1.jpg"
                    path=os.path.join(self.image_root, image_name)
                    self.append_id_repeat.append(image_id)
                    self.repeat_name.append(image_name)
        
            ## Append the new image name. 
            else: 
                self.append_id.append(image_id)
                self.append_id_repeat.append(image_id)
                self.repeat_name.append(image_name)

            with torch.autocast('cuda'):
                            generate_image = self.model(
                                prompt=[caption],
                                mode="prompt",
                                height=512,
                                width=512,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                init_image=None,
                                generator=self.generator,
                                strength=0.8,
                                return_intermediates=False,
                            )['sample']
        generate_image[0].save(path)
        self.append_id.append(image_id)
        print(f"image name {image_id} Generated")
        return image_name

    def save_json(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.new_json, outfile)

class synthetic_text_2_image(Dataset):

    def __init__(self, image_root, ann_root, max_words=200, prompt='4k , highly detailed', generate_mode="repeat", 
                         guidance_scale=7.5,  num_inference_steps=50, seed=123245):
        '''
        image_root (string): Root directory for storing the generated images (ex: /data/coco_synthetic/)
        anno_root(string): directory for storing the human caption file from COCO Caption dataset
        '''
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'

        Path(image_root + "val2014/").mkdir(parents=True, exist_ok=True)
        Path(image_root + "train2014/").mkdir(parents=True, exist_ok=True)
        Path(ann_root).mkdir(parents=True, exist_ok=True)

        # os.makedirs(image_root+ "val2014/", exist_ok=True)
        # os.makedirs(ann_root, exist_ok=True)

        download_url(url, ann_root)
        self.annotation = json.load(
            open(os.path.join(ann_root, filename), 'r'))

        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generate_mode= generate_mode
        # random.randint(0, 100000) #random.randint(0,10000) # change the seed to get different results
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        self.append_id = ["test"]
        self.repeat_name=["test"]
        self.append_id_repeat=["test"]
        
        
        ### Stable Diffusion 1.4
        store_path="/data1/pretrained_weight/StableDiffusion/"
        # self.model = StableDiffusionPipeline.from_pretrained(
        #     "CompVis/stable-diffusion-v1-4", revision="fp16",
        #     torch_dtype=torch.float32,
        #     use_auth_token=True,
        #     cache_dir= store_path, 
        # ).to("cuda")

        ### Stable Diffusion 2.1
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="fp16",#stabilityai/stable-diffusion-2-1
            torch_dtype=torch.float16,
            use_auth_token=True,
            cache_dir= store_path, 
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.model=pipe.to("cuda")
        

        
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        ann = self.annotation[idx]

        caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        image_id = ann['image_id']
        image_name = ann['image']  # Saved image's name
        path=os.path.join(self.image_root, image_name)
        if self.generate_mode=="repeat":
            if image_id == self.append_id[-1]:
                #print("Using mode Image to generate image")
                init_image = Image.open(os.path.join(self.image_root, image_name))
                with torch.autocast('cuda'):
                    generate_image = self.model(
                        prompt=[caption],
                        mode="image",
                        height=512,
                        width=512,
                        num_inference_steps=50,
                        guidance_scale=self.guidance_scale,
                        init_image=init_image,
                        generator=self.generator,
                        strength=0.8,
                        return_intermediates=False,
                    ).images[0]

            else:  # Case not repeat image
                #print("Using mode Prompt to generate image")
                        # with torch.autocast('cuda'):
                        #     generate_image = self.model(
                        #         prompt=[caption],
                        #         mode="prompt",
                        #         height=512,
                        #         width=512,
                        #         num_inference_steps=self.num_inference_steps,
                        #         guidance_scale=self.guidance_scale,
                        #         init_image=None,
                        #         generator=self.generator,
                        #         strength=0.8,
                        #         return_intermediates=False,
                        #     ).images[0]

                        with torch.autocast('cuda'):  
                            generate_image = self.model(
                                prompt=[caption],
                                height=512,
                                width=512,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                generator=self.generator,
                            ).images[0]

        else: 
            ## inCase the image name repeat 
            if image_id == self.append_id[-1]:
                # The first image repeat is creat
                image_name_= self.repeat_name[-1][:-5] + "1" +".jpg"
                path=os.path.join(self.image_root, image_name_)

                #checking image is exist or not
                if os.path.isfile(path) is True or os.path.exists(path) is True:
                    print("Next repeat image is append")
                    image_name= self.repeat_name[-1][:-4] + "1" + ".jpg"
                    image_id=self.append_id_repeat[-1] + "1"
                    path=os.path.join(self.image_root, image_name)
                    self.append_id_repeat.append(image_id)
                    self.repeat_name.append(image_name)

                else:
                    print("first repeat image is created")
                    image_id= image_id +"1"
                    image_name= image_name[:-4] + "1.jpg"
                    path=os.path.join(self.image_root, image_name)
                    self.append_id_repeat.append(image_id)
                    self.repeat_name.append(image_name)
        
            ## Append the new image name. 
            else: 
                self.append_id.append(image_id)
                self.append_id_repeat.append(image_id)
                self.repeat_name.append(image_name)

            with torch.autocast('cuda'):  
                generate_image = self.model(
                    prompt=[caption],
                    height=512,
                    width=512,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=self.generator,
                ).images[0]

        generate_image.save(path)
        self.append_id.append(image_id)
        print(f"image name {image_id} Generated")
        return image_name

    def save_json(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.new_json, outfile)

# generate_data= COCO_synthetic_Dataset(image_root='/data1/coco_synthetic/coco_synthetic/', ann_root='/data1/coco_synthetic/', generate_mode="no_repeat")
# ## CoCo Caption dataset Caption Length 566.747=
# print(generate_data.__len__())
# for i in range(400000, 500000):
#     generate_data.__getitem__(i)
# generate_data.save_json("/data1/coco_synthetic/coco_synthetic/coco_synthetic_400k_500k.json") 
# print("------------------------ Done ------------------------")

## ------------------------ 1.1 TEXT to IMAGE Guided CLIP Vision -------------------------------
from transformers import CLIPFeatureExtractor, CLIPModel
from diffusers import DiffusionPipeline 

class text_image_clip_guided_SD(Dataset): 
    '''
    Several Approach Using CLIP Guided Image Generation
    '''
    def __init__(self,image_root, ann_root, max_words=200, prompt='4k , highly detailed', 
                         guidance_scale=7.5,  num_inference_steps=50, seed=123245 ):
        
        
        ## Setting up the model
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.float16)
        store_path="/data1/pretrained_weight/StableDiffusion/"
      
        pipe= DiffusionPipeline.from_pretrained(
                                                "runwayml/stable-diffusion-v1-5",
                                                #"CompVis/stable-diffusion-v1-4",
                                                custom_pipeline= "clip_guided_stable_diffusion", 
                                                clip_model=self.clip_model,
                                                clip_feature_extractor=self.feature_extractor,
                                                torch_dtype=torch.float16, 
                                                cache_dir= store_path,
                                                # scheduler=scheduler  
                                                )
       
        #pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.guided_pipeline= pipe.to("cuda")

        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'

        Path(image_root + "val2014/").mkdir(parents=True, exist_ok=True)
        Path(image_root + "train2014/").mkdir(parents=True, exist_ok=True)
        Path(ann_root).mkdir(parents=True, exist_ok=True)

        # os.makedirs(image_root+ "val2014/", exist_ok=True)
        # os.makedirs(ann_root, exist_ok=True)

        download_url(url, ann_root)
        self.annotation = json.load(
            open(os.path.join(ann_root, filename), 'r'))

        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        # random.randint(0, 100000) #random.randint(0,10000) # change the seed to get different results
        self.generator = torch.Generator(device="cuda").manual_seed(seed)

        ## Create a set to store used image IDs
        self.id_count = {}
        self.new_annotation=[]
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):

        ann = self.annotation[idx]
        caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        image_id = ann['image_id']
        image_name = ann['image']  # Saved image's name
        path=os.path.join(self.image_root, image_name)
        
        if image_id in self.id_count:
            count = self.id_count[image_id]
            image_id_ = f"{image_id}{count}"
            image_name = f"{image_name[:-4]}{count}.jpg"
            path = os.path.join(self.image_root, image_name)
            self.id_count[image_id] += 1
        else: 
            self.id_count[image_id] = 1
            image_id_=image_id
        ## inCase the image name repeat 
      
        with torch.autocast('cuda'):  
            generate_image = self.guided_pipeline(
                prompt=[caption],
                height=512,
                width=512,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=self.generator,
            ).images[0]
        new_annotation = {'caption': ann['caption'] , 'image': image_name,'image_id':image_id_  }
        self.new_annotation.append(new_annotation)
        generate_image.save(path)
 
        print(f"Image with id '{image_id}' generated")
        #print(f"image name {image_id} Generated")
        return image_name

    def save_json(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.new_annotation, outfile)

# generate_data= text_image_clip_guided_SD(image_root='/data1/DataEngine/TextImageClipGuidedSD/', ann_root='/data1/original_coco/')
# ## CoCo Caption dataset Caption Length 566.747=
# print(generate_data.__len__())
# for i in range(400000,  566747):
#     generate_data.__getitem__(i)
# generate_data.save_json("/data1/DataEngine/TextImageClipGuidedSD/coco_synthetic_400k_566747k.json") 
# print("------------------------ Done ------------------------")

class synthetic_text_2_image_Dalle_SD(Dataset): 

    def __init__(self, image_root, ann_root, max_words=200, prompt='A photo of highly detailed of ', 
                    dalle_topk=128, temperature=2., supercondition_factor=16.,
                    guidance_scale=7.5,  num_inference_steps=50, seed=random.randint(50000, 1000000)):
        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'
        Path(image_root+ "val2014/").mkdir(parents=True, exist_ok=True)
        Path(image_root+ "train2014/").mkdir(parents=True, exist_ok=True)
        Path(image_root+ "dalle_image/").mkdir(parents=True, exist_ok=True)

        Path(ann_root).mkdir(parents=True, exist_ok=True)
        download_url (url, ann_root)
        self.annotation= json.load(open(os.path.join(ann_root, filename), 'r'))

        self.image_root= image_root 
        self.max_words= max_words
        self.prompt= prompt
        ## Parameter for Dalle-Mini Model
        self.dalle_topk=dalle_topk
        self.temeperature= temperature 
        self.supercondition_factor= supercondition_factor 
        self.Dalle_model = MinDalle(
        # /home/rick/pretrained_weights/Dalle_mini_mega
            models_root="/data1/pretrained_weight/Dalle_mini_mega",
            dtype=torch.float32,
            is_mega=False,  # False -> Using mini model,
            device='cuda',
            is_reusable=True,)

        ## Parameter for StableDiffusion Model 
        self.guidance_scale= guidance_scale
        self.num_inference_steps= num_inference_steps
        self.generator = torch.Generator(device="cuda").manual_seed(seed) #random.randint(0, 100000) #random.randint(0,10000) # change the seed to get different results
        self.append_id=["test"]
        self.append_id_repeat=["test"]
        self.repeat_name=['test']

        self.SD_model= StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,).to("cuda")
        self.new_json=[]
    
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        ann= self.annotation[idx]
        
        caption= self.prompt + pre_caption(ann['caption'], self.max_words)
        image_id= ann['image_id']
        image_name= ann['image'] ## Saved image's name
        
        ## Caption, value, name, value, image_name, value
        path=os.path.join(self.image_root, image_name)
   
        ## inCase the image name repeat 
        if image_id == self.append_id[-1]:
            # The first image repeat is creat
            image_name_= self.repeat_name[-1][:-5] + "1" +".jpg"
            path=os.path.join(self.image_root, image_name_)

            #checking image is exist or not
            if os.path.isfile(path) is True or os.path.exists(path) is True:
                print("Next repeat image is append")
                image_name= self.repeat_name[-1][:-4] + "1" + ".jpg"
                image_id=self.append_id_repeat[-1] + "1"
                path=os.path.join(self.image_root, image_name)
                self.append_id_repeat.append(image_id)
                self.repeat_name.append(image_name)

            else:
                print("first repeat image is created")
                image_id= image_id +"1"
                image_name= image_name[:-4] + "1.jpg"
                path=os.path.join(self.image_root, image_name)
                self.append_id_repeat.append(image_id)
                self.repeat_name.append(image_name)
    
        ## Append the new image name. 
        else: 
            self.append_id.append(image_id)
            self.append_id_repeat.append(image_id)
            self.repeat_name.append(image_name)
        
        with torch.autocast('cuda'):
                init_image= self.Dalle_model.generate_image(text=caption,
                                                seed=-1,
                                                grid_size=1,
                                                is_seamless=False,  # If this set to False --> Return tensor
                                                temperature=self.temeperature,
                                                top_k=self.dalle_topk,
                                                supercondition_factor=self.supercondition_factor,
                                                is_verbose=False
                                                ) 
                print(init_image.size)
                init_image = init_image.resize((512, 512))
                generate_image= self.SD_model(
                                prompt=[caption],
                                mode="image",
                                height=512,
                               width=512,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                init_image=init_image,
                                generator=None, #self.generator,
                                strength=0.8,
                                return_intermediates=False,
                                )['sample']

       
        generate_image[0].save(path)
        #init_image.save(os.path.join("/data1/coco_synthetic/dalle_image/"+image_name))
        new_dict={"caption":ann['caption'],"image": image_name, "image_id": image_id}  
        self.new_json.append(new_dict)
        print(f"image name: {caption} Generated")
        return image_name 
        
    def save_json(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.new_json, outfile)

# generate_data= COCO_synthetic_Dalle_SD(image_root='/data1/coco_synthetic_Dalle_SD/', ann_root='/data1/coco_synthetic_Dalle_SD/')
# for i in range(150000, 200000):
#     generate_data.__getitem__(i)
# print("------------------------ Done ------------------------")
# generate_data.save_json("/data1/coco_synthetic_Dalle_SD/coco_synthetic_150k_200k.json")

## ----------------------- 2--- IMAGE VARIANTS ------------------------
     
class Synthetic_Img_invariance(torch.utils.data.Dataset):
  def __init__(self, data_dir, output_dir, transform=None, num_steps=35, guidance_scale=3,num_img_invariant=5, ):
    self.data_dir = data_dir
    with open(os.path.join(data_dir,  "coco_karpathy_train.json"), 'r') as f:
      self.captions = json.load(f)


    self.output_dir = output_dir
    self.num_steps=num_steps
    self.num_img_invariant=num_img_invariant
    self.guidance_scale=guidance_scale
    
    if transform is None:
        self.transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (512, 512),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]),
        ])


    else: 
        self.transform = transform

    self.image_ids = set()  # Set to store unique image IDs
    
    ## Stable Diffusion Model with Image Variant Model 
    store_path="/data1/pretrained_weight/StableDiffusion/"
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
                "lambdalabs/sd-image-variations-diffusers",
                revision="v2.0",
            torch_dtype=torch.float16,
            use_auth_token=True,
            cache_dir= store_path, 
            
        )
    sd_pipe.safety_checker = dummy
    sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
    self.generative_model=sd_pipe.to("cuda")

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, index):
    
    ## Get unique image id
    # new_captions = []
    while True:
        annotation = self.captions[index]
        image_id = annotation['image']
        # id= annotation['image_id']
        # caption= annotation['caption']
        # new_captions.append({'image_id':id, 'caption':caption})

        if image_id not in self.image_ids:  # Check if image has been loaded already
            self.image_ids.add(image_id)  # Add image to set of loaded images
        break
    
    index = (index + 1) % len(self.captions)  # Increment index and wrap around if necessary

    path = os.path.join(self.data_dir,image_id)
   
    ## Load Image
    # image = io.imread(path)
    
    init_img = Image.open(path)

    # Check if the image is grayscale
    if init_img.mode == "L":
        # Convert the grayscale image to an RGB image
        init_img = init_img.convert("RGB")

    init_img= init_img.resize((512, 512),resample=Image.BILINEAR)
    
    ## Transform Image
    # img = self.transform(img).to(device)#.unsqueeze(0)


    ## 2... Generate 5 Invariant Images from 1 Image   
    num_samples = self.num_img_invariant
    image = self.generative_model(num_samples*[init_img],num_inference_steps=self.num_steps, guidance_scale=self.guidance_scale).images
    Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    
    output_path_=os.path.join(self.output_dir, image_id[:-4])
    for idx, im in enumerate(image):
        output_path=output_path_ + str("invar_")+ str(idx) + ".jpg"
        #output_path=os.path.join(output_path,str(idx) + ".jpg")
        im.save(output_path)
        
    
    # invariants = []
    # for i in range(5):
    #   invariant = self.generative_model.generate_invariant(image)
    #   invariants.append(invariant)
    #   io.imsave

    def save_json(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.new_json, outfile)

# data_dir="/data1/original_coco/"
# generate_data=Synthetic_Img_invariance(data_dir=data_dir, output_dir="/data1/coco_SD_invariant_synthetic",)
# with open(os.path.join(data_dir,  "coco_karpathy_train.json"), 'r') as f:
#     captions = json.load(f)

# for i in range(450000,566747 ):
#     generate_data.__getitem__(i)
# print("======================== Done ========================")
# generate_data.save_json("/data1/coco_synthetic_Dalle_SD/coco_synthetic_150k.json")


## ----------------------- 3--- IMAGE 2 IMAGE ------------------------
class synthetic_Image_Depth_image(torch.utils.data.Dataset): 
    
    def __init__(self, data_dir, output_dir, transform=None, num_steps=50, guidance_scale=7.5, ):
        self.data_dir = data_dir

        Path(output_dir + "/val2014/").mkdir(parents=True, exist_ok=True)
        Path(output_dir + "/train2014/").mkdir(parents=True, exist_ok=True)

        with open(os.path.join(data_dir,  "coco_karpathy_train.json"), 'r') as f:
            self.annotations = json.load(f)

        self.output_dir = output_dir
        self.num_steps=num_steps
        self.guidance_scale=guidance_scale
        self.new_json=[] 


        ## Stable Diffusion Model with Image Variant Model 
        store_path="/data1/pretrained_weight/StableDiffusion/"
        sd_pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-depth",
                torch_dtype=torch.float16,
                use_auth_token=True,
                cache_dir= store_path, 
                
            )
        sd_pipe.safety_checker = dummy
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        self.generative_model=sd_pipe.to("cuda")
    
    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        annotations = self.annotations[index]
        ## Get image information
        image_id = annotations['image']
        caption = annotations['caption']
        
        ## Reading initial image 
        path = os.path.join(self.data_dir,image_id)
        init_img = Image.open(path)

        # Check if the image is grayscale
        if init_img.mode == "L":
            # Convert the grayscale image to an RGB image
            init_img = init_img.convert("RGB")

        init_img= init_img.resize((512, 512),resample=Image.BILINEAR)
        
        ## Generate new image with input prompt
        generator=torch.Generator(device="cuda").manual_seed(random.randint(0,100000))

        ## Save image
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # check_path=os.path.join(self.output_dir, image_id[:-4] + f"_0_SD_depth.jpg")
        # if os.path.exists(check_path):
        #     print("The image file exists.")
        # else:
        #     print("The image file does not exist.")
        i = 0
        while True:
            output_path = os.path.join(self.output_dir, image_id[:-4] + f"_{i}_SD_depth.jpg")
            if not os.path.exists(output_path):  # Check if file with same name already exists
                break
            i += 1  # Increment counter and try next name
        
        image = self.generative_model(prompt=caption, image=init_img,  num_inference_steps=self.num_steps,  guidance_scale=self.guidance_scale, strength=0.7, generator=generator).images[0]
        image.save(output_path)

        ## Save new image_id and caption
        new_image_id = image_id[:-4] + f"_{i}_SD_depth.jpg"
        ID= annotations['image_id']
        new_id = ID + f"_{i}_SD_depth"
        
        new_annotation = {'caption': caption , 'image': new_image_id,'image_id':new_id  }
        self.new_json.append(new_annotation)

    def save_json(self, file_name):
        with open(os.path.join(self.output_dir,file_name), 'w') as f:
            json.dump(self.new_json, f)

# generate_data=synthetic_Image_Depth_image(data_dir="/data1/original_coco_caption/", output_dir="/data1/coco_synthetic/coco_synthetic_img_depth_img/",)
# for i in range(4500000, ):
#     # print(i)

#     generate_data.__getitem__(i)
# print("======================== Done ========================")
# generate_data.save_json("coco_synthetic_450k_566747k.json")
## 227 sever missing Json file from (150k-300k)


## ----------------------- 3--- IMAGE CAPTIONING ------------------------
# from transformers import BlipProcessor, AutoProcessor, BlipForConditionalGeneration
# from lavis.models import load_model_and_preprocess
from transformers import BlipProcessor, AutoProcessor, BlipForConditionalGeneration, Blip2ForConditionalGeneration
class COCO_Synthetic_BLIP_Image_Caption(Dataset): 
    # Generate Caption as following
    # 1. Using Original CoCo Image --> Generate synthetic captions 
    # 2. Using Synthetic Image --> Generate synthetic Caption

    def __init__(self,data_dir, json_file_name, output_dir, output_json_file, blip_2=False ):
        
        ## Reading Image --> Preprocess image --> Conditional Captioning --> Synthetic Captioning. 
        # Creating a store path for Synthetic Captioning 
  
        #filename = 'coco_karpathy_train.json'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        ## Get the original coco annotation to get image ID, image path
        self.coco_dataset = json.load(
            open(os.path.join(data_dir, json_file_name), 'r'))
        
        # with open(os.path.join(data_dir, json_file_name ), 'r') as f:
        #     self.coco_dataset = json.load(f)

        ## Get BLIP model
        store_path="/data1/pretrained_weight/BLIP/"
        Path(store_path).mkdir(parents=True, exist_ok=True)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", 
                                                       torch_dtype=torch.float16, 
                                                       cache_dir= store_path, 
                                                       )
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", 
                                                                    torch_dtype=torch.float16, 
                                                                    cache_dir= store_path, 
                                                                    ).to("cuda")

        # Get BLIP Model V2 
        self.blip2_processor= AutoProcessor.from_pretrained(
                                                            'salesforce/blip2-opt-6.7b',
                                                            # Salesforce/blip2-flan-t5-xxl    
                                                            torch_dtype=torch.float16,
                                                            cache_dir= store_path,
              )
        #self.blip2_model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)

        self.blip_2_model= Blip2ForConditionalGeneration.from_pretrained(
                                                            'salesforce/blip2-opt-6.7b', 
                                                            # Salesforce/blip2-flan-t5-xxl   
                                                            torch_dtype=torch.float16, 
                                                            cache_dir= store_path,
                                                            ).to("cuda")

        self.data_dir=data_dir
        self.output_json_file = output_json_file
        self.blip2=blip_2
        # New synthetic annotation will save in new_json
        self.output_json_file = output_json_file
        self.captions = []
        self.processsed_image_ids=set()
        self.not_found_image_ids=set()

    ## Incase want to iterate through certain number of images
    # def __len__(self):
    #     unique_image_ids = set()
    #     for item in self.coco_dataset:
    #         if item['image_id'] not in unique_image_ids:
    #             unique_image_ids.add(item['image_id'])

    #     return len(unique_image_ids)

    ## Iterate through all the images
    def __len__(self):

        return len(self.coco_dataset)
        

    def __getitem__(self, idx):
        #print(len(self.coco_dataset))
        image_info = self.coco_dataset[idx]
        image_id = image_info['image_id']
        image_name= image_info['image']
        image_path = os.path.join(self.data_dir, image_info['image'])
        #breakpoint()

        if image_id in self.processsed_image_ids: 
            return None
        
        #print(image_path)
        if os.path.isfile(image_path):
            raw_image = Image.open(image_path).convert('RGB')
        else: 
            print("Error: Could not find the file at the specified path.")
            self.not_found_image_ids.add(image_id)
            return None
    
        
        if self.blip2: 
            text = "a photography of"
            inputs= self.blip2_processor(raw_image,text, return_tensors="pt", ).to("cuda", torch.float16)
            out_1= self.blip_2_model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
            out_2= self.blip_2_model.generate(**inputs, max_length=100, do_sample=True, top_k=50, top_p=0.8, num_return_sequences=3)
            out_3= self.blip_2_model.generate(**inputs, max_length=100, penalty_alpha=0.6, top_k=4)
            beam_caption = self.blip2_processor.decode(out_1[0], skip_special_tokens=True)
            nucleus_caption = self.blip2_processor.decode(out_2[0], skip_special_tokens=True)
            contrastive_caption = self.blip2_processor.decode(out_3[0], skip_special_tokens=True)

            self.captions.append({"image_name": image_name, 'image_id': image_id, 'beam_caption': beam_caption, 'nucleus_caption': nucleus_caption, "contrastive_caption": contrastive_caption})
            self.processsed_image_ids.add(image_id)
            #breakpoint()
            return image_info, beam_caption, nucleus_caption, contrastive_caption
        
        else: 
            print("Using BLIP1")
            # conditional image captioning
            text = "a photography of"
            inputs = self.processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)
            
            strategy = "beam_search"
            #if strategy == "beam_search":
            out_1 = self.model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
            #elif strategy=="nucleus_sampling":
            out_2 = self.model.generate(**inputs, max_length=100, do_sample=True, top_k=50, top_p=0.8, num_return_sequences=3)
            #elif strategy =="contrastive_search": 
            out_3 = self.model.generate(**inputs, max_length=100, penalty_alpha=0.6, top_k=4)
            #out = self.model.generate(**inputs)

            beam_caption = self.processor.decode(out_1[0], skip_special_tokens=True)
            beam_caption = beam_caption.replace("a photography of", "")
            nucleus_caption = self.processor.decode(out_2[0], skip_special_tokens=True)
            nucleus_caption = nucleus_caption.replace("a photography of", "")
            contrastive_caption = self.processor.decode(out_3[0], skip_special_tokens=True)
            contrastive_caption = contrastive_caption.replace("a photography of", "")
            # save the captions and image id
            self.captions.append({"image_name": image_name, 'image_id': image_id, 'beam_caption': beam_caption, 'nucleus_caption': nucleus_caption, "contrastive_caption": contrastive_caption})
            self.processsed_image_ids.add(image_id)
            #breakpoint()
            return image_info, beam_caption, nucleus_caption, contrastive_caption                               

    def save_to_json(self):
        path= os.path.join(self.output_dir, self.output_json_file)
        with open(path, 'w') as f:
            json.dump(self.captions, f)

        ## Save the image ids that are not found
        path_= os.path.join(self.output_dir, "image_ids_not_found.json")
        not_foud_image_ids=list(self.not_found_image_ids)
        with open(path_, 'w') as f:
            json.dump(not_foud_image_ids, f)

generate_data=COCO_Synthetic_BLIP_Image_Caption(data_dir="/data1/coco_SD_invariant_synthetic/", json_file_name="coco_SD_invariant_synthetic.json", 
                                                output_dir="/data1/DataEngine/blip_clip_sd_images_caption_synth/",output_json_file="BLIP2_Clip_SD_image_to_text.json", blip_2=True)

## For Loop To Generate Synthetic Caption
## Total Unique Image ID = 113287
## Totoal Image ID = 566747
for i in tqdm(range(566435)):
    generate_data.__getitem__(i)
generate_data.save_to_json()
print("======================== Done ========================")

class GIT_synthetic_image_caption(Dataset):
    pass 

# 227 sever missing Json file from (150k-300k)
class COCO_Synthetic_VQA(Dataset): 
    pass