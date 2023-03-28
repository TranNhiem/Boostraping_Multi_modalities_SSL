import os
import torch
from torchvision import transforms
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CityscapesSegmentation(Dataset):
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
## iterate through the dataset to get images and labels
for i, (image, label) in enumerate(dataset):
    # do something with the image and label, for example print their shapes
    print(f"Image shape: {image.size}, Label shape: {label.size}")
    label.save('test_segment.png')
    image.save('test_image.png')
    break


