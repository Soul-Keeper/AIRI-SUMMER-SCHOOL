import cv2
import json
import torch
import numpy as np
import torchvision

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score
from transformers import AutoModel, AutoTokenizer

image_path = "../data/vto+color/train/"
annotations_path = "../data/vto+color/train/_annotations.coco.json"

f = open(annotations_path)
annotations = json.load(f)

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')
model.eval()

print(f"Model size: {model.num_parameters()}")

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)

iou = []

gt_cls = []
pred_cls = []


for record in tqdm(annotations["images"][:1]):
    try:
        id = record['id']
        #box = annotations["annotations"][id]['bbox']
        #gt_box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        #gt_box = torch.tensor([[box[0], box[1], box[0] + box[2], box[1] + box[3]]], dtype=torch.float)
        ground_truth_color = record['extra']['user_tags'][1].split('-')[1]
        #ground_truth_cls = record['extra']['user_tags'][0].split('-')[1]
        #image = Image.open(image_path + record['file_name']).convert('RGB')
        image = Image.open('/home/user8/nsidelnikov/data/kroy/train/images/10_jpg.rf.568fc9f9aa33cd0799b9b18440b6a10e.jpg').convert('RGB')
        # image_cv = cv2.imread(image_path + record['file_name'])

        question = 'The photo shows patterns on the surface of the fabric. Provide a segmentation mask of the patterns.'
        msgs = [{'role': 'user', 'content': question}]

        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True, # if sampling=False, beam_search will be used by default
            temperature=0.7,
            # system_prompt='' # pass system_prompt if needed
        )
        print(res)
        
        gt_cls.append(ground_truth_color)
        pred_cls.append(res.split(' ')[-1])
        
        #print(ground_truth_color, res.split(' ')[-1])
        
        #bbox = res[5:-6].replace('<', '').split(' ')
        #pred_box = [int(n) for n in bbox]
        #pred_box = torch.tensor([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=torch.float)
        
        #iou.append(torchvision.ops.boxes.box_iou(gt_box, pred_box).item())

        #print(gt_box)
        #print(pred_box)     
        
        #image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        #image_cv = cv2.rectangle(image_cv, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, ), 2)
        #image_cv = cv2.rectangle(image_cv, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 0, 255), 1)
        #cv2.imshow('test', image_cv)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    except Exception as e: 
        print(id, e)
    
#print("IoU: ", np.mean(iou))
print("cls acc: ", accuracy_score(gt_cls, pred_cls))