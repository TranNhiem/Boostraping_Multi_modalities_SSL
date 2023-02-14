# import os
# from PIL import Image

# image_path = "/data1/coco_synthetic/train2014/COCO_train2014_000000016653.jpg"

# # try:
# #     raw_image = Image.open(image_path).convert('RGB')
# # except FileNotFoundError:
# #     print("Error: Could not find the file at the specified path.")
    
# if os.path.isfile(image_path):
#     raw_image = Image.open(image_path).convert('RGB')
# else:
#     print("Error: Could not find the file at the specified path.")
    
#     # Add code here to continue running the rest of the program.
# print("Success: The file was found at the specified path.")


## Function to reading COCO annotation file and return a dictionary of image_id and its corresponding annotation
import json
import os 
def get_coco_annotation(coco_annotation_path, save_dir="/data1/"):
    '''
    input : coco_annotation_path, 
    output: new coco_annotation_dict contain image_id and its variant 4 image_ids from each original ID

    '''
    coco_SD_invariant_synthetic =[]
    processsed_image_ids=set()
    with open(coco_annotation_path, 'r') as f:
        coco_annotation = json.load(f)
        for i in range(len(coco_annotation)):

            single_dict= coco_annotation[i]
            
            ## Consider to save image path and Image Name 
            image_id = single_dict['image_id']
            image= single_dict['image']
            if image_id in processsed_image_ids:    
                continue
            else:
                processsed_image_ids.add(image_id)
                for i in range(5): # 5 is the number of variant
                    image_id = image_id+str(i)
                    ## To save new annotation
                    image=image[:-4]+ str(i) + ".jpg"
                    new_json={"image_id": image_id, "image": image}
                    
                    coco_SD_invariant_synthetic.append(new_json)
                #breakpoint()
                ## To save new annotation
                #new_json={"image_id": image_id, "image": image}
                #coco_SD_invariant_synthetic.append(new_json)
    
                output_json_file= "coco_SD_invariant_synthetic.json"
    path= os.path.join(save_dir, output_json_file)
    with open(path, 'w') as f:
        json.dump(coco_SD_invariant_synthetic, f)
    return coco_SD_invariant_synthetic

get_image_id= get_coco_annotation("/data1/original_coco/coco_karpathy_train.json", save_dir="/data1/coco_SD_invariant_synthetic/")

def get_coco_annotation_2(coco_annotation_path):
    '''
    input : coco_annotation_path, 
    output: new coco_annotation_dict contain image_id and its variant 4 image_ids from each original ID

    '''
    coco_annotation_dict = {}
    with open(coco_annotation_path, 'r') as f:
        coco_annotation = json.load(f)
        for i in range(len(coco_annotation['images'])):
            coco_annotation_dict[coco_annotation['images'][i]['file_name']] = coco_annotation['images'][i]['id']
    return coco_annotation_dict