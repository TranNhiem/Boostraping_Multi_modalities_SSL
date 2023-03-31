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


# ## Function to reading COCO annotation file and return a dictionary of image_id and its corresponding annotation
# import json
# import os 
# def get_coco_annotation(coco_annotation_path, save_dir="/data1/"):
#     '''
#     input : coco_annotation_path, 
#     output: new coco_annotation_dict contain image_id and its variant 4 image_ids from each original ID

#     '''
#     coco_SD_invariant_synthetic =[]
#     processsed_image_ids=set()
#     with open(coco_annotation_path, 'r') as f:
#         coco_annotation = json.load(f)
#         for i in range(len(coco_annotation)):

#             single_dict= coco_annotation[i]
            
#             ## Consider to save image path and Image Name 
#             image_id = single_dict['image_id']
#             image_= single_dict['image']
#             if image_id in processsed_image_ids:    
#                 continue
#             else:
#                 processsed_image_ids.add(image_id)
#                 for i in range(5): # 5 is the number of variant
#                     image_id = image_id+str(i)
#                     ## To save new annotation
#                     image=image_[:-4]+ "invar_"+ str(i) + ".jpg"

#                     new_json={"image_id": image_id, "image": image}
                    
#                     coco_SD_invariant_synthetic.append(new_json)
#                 #breakpoint()
#                 ## To save new annotation
#                 #new_json={"image_id": image_id, "image": image}
#                 #coco_SD_invariant_synthetic.append(new_json)
    
#                 output_json_file= "coco_SD_invariant_synthetic.json"
#     path= os.path.join(save_dir, output_json_file)
#     with open(path, 'w') as f:
#         json.dump(coco_SD_invariant_synthetic, f)
#     return coco_SD_invariant_synthetic

# get_image_id= get_coco_annotation("/data1/original_coco/coco_karpathy_train.json", save_dir="/data1/coco_SD_invariant_synthetic/")

# def get_coco_annotation_2(coco_annotation_path):
#     '''
#     input : coco_annotation_path, 
#     output: new coco_annotation_dict contain image_id and its variant 4 image_ids from each original ID

#     '''
#     coco_annotation_dict = {}
#     with open(coco_annotation_path, 'r') as f:
#         coco_annotation = json.load(f)
#         for i in range(len(coco_annotation['images'])):
#             coco_annotation_dict[coco_annotation['images'][i]['file_name']] = coco_annotation['images'][i]['id']
#     return coco_annotation_dict

## Testing OpenAI API 
import os
import openai

# # Replace "your_api_key" with your actual API key
# openai.api_key = "0aee54a3f2df4c55aea57bf3cf2e99a6"

# def generate_text(prompt, model="text-davinci-002", max_tokens=50):
#     response = openai.Completion.create(
#         engine=model,
#         prompt=prompt,
#         max_tokens=max_tokens,
#         n=1,
#         stop=None,
#         temperature=0.7,
#     )
#     return response.choices[0].text.strip()

# if __name__ == "__main__":
#     prompt = "Once upon a time in a land far, far away,"
#     generated_text = generate_text(prompt)
#     print(generated_text)


import os
import openai
openai.api_type = "azure"
openai.api_base = "https://sslgroupservice.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
#openai.api_key =  "97f22a7a32ff4ff4902003896f247ca2"
#openai.api_key = os.environ["OPENAI_API_KEY"]

response = openai.ChatCompletion.create(
  engine="gpt-35-turbo",
    messages=[{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "what is the capital of California?"},],
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

print(response)