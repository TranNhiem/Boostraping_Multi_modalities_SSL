import os 
from PIL import Image 
from torchvision import transforms

# Set the path to the image file
image_path = "/data1/original_coco_caption/train2014/COCO_train2014_000000061048.jpg"

# Check if the image file exists
if os.path.exists(image_path):
    print("The image file exists.")
    image= Image.open(image_path)
    
    init_img= image.resize((512, 512),resample=Image.BILINEAR)
    print(init_img.size)


    transform= transforms.Compose([
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
    img= transform(init_img)
    print(img.shape)
else:
    print("The image file does not exist.")