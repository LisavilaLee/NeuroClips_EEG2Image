'''
Description: 
Author: Zhou Tianyi
Date: 2025-02-08 08:20:57
LastEditTime: 2025-02-20 07:31:22
LastEditors:  
'''
from PIL import Image
from torchvision import transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import numpy as np

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
proxy = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b',torch_dtype=torch.float16)
model.to(device)

images = torch.load('/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/Models/EEG/EEG2Video/EEG2Video2/neuroclips/frames_generated/glfnet_ideo_subj01_SR_use_label_use_labels_/glfnet_ideo_subj01_SR_use_label_use_labels__all_recons_way.pt',map_location='cpu')
print(images.shape)
for i in range(images.shape[0]):
    x = images[i]
    x = transforms.ToPILImage()(x)
    inputs = processor(images=x, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    all_predcaptions = np.hstack((all_predcaptions, generated_text))
    print(generated_text, all_predcaptions.shape)
    exit()

torch.save(all_predcaptions, f'/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/Models/EEG/EEG2Video/EEG2Video2/neuroclips/frames_generated/glfnet_ideo_subj01_SR_use_label_use_labels_/video_subj01_blip_caption.pt')
# images = torch.load('/fs/scratch/PAS2490/neuroclips/datasets--gongzx--cc2017_dataset/snapshots/a82b9e20e98710f18913a10c0a5bf5f19a6e4000/GT_test_3fps.pt',map_location='cpu')[:,2,:,:,:]
# print(images.shape)
# all_predcaptions = []
# for i in range(images.shape[0]):
#     x = images[i]
#     x = transforms.ToPILImage()(x)
#     inputs = processor(images=x, return_tensors="pt").to(device, torch.float16)

#     generated_ids = model.generate(**inputs)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     all_predcaptions = np.hstack((all_predcaptions, generated_text))
#     print(generated_text, all_predcaptions.shape)

# torch.save(all_predcaptions, f'/fs/scratch/PAS2490/neuroclips/GT_test_caption.pt')


# images = torch.load('/fs/scratch/PAS2490/neuroclips/datasets--gongzx--cc2017_dataset/snapshots/a82b9e20e98710f18913a10c0a5bf5f19a6e4000/GT_train_3fps.pt',map_location='cpu')[:,2,:,:,:]
# print(images.shape)
# all_predcaptions = []
# for i in range(images.shape[0]):
#     x = images[i]
#     x = transforms.ToPILImage()(x)
#     inputs = processor(images=x, return_tensors="pt").to(device, torch.float16)

#     generated_ids = model.generate(**inputs)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     all_predcaptions = np.hstack((all_predcaptions, generated_text))
#     print(generated_text, all_predcaptions.shape)

# torch.save(all_predcaptions, f'/fs/scratch/PAS2490/neuroclips/GT_train_caption.pt')