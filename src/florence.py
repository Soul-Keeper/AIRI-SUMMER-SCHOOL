import cv2
import json
import torch
import torchvision
import numpy as np

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM  

def run_example(image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.shape[1], image.shape[0])
    )

    return parsed_answer


model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print(f"Model size: {model.num_parameters()}")

image_path = "../data/vto+color/train/"
annotations_path = "../data/vto+color/train/_annotations.coco.json"

f = open(annotations_path)
annotations = json.load(f)

task_prompt = '<OPEN_VOCABULARY_DETECTION>'

iou = []

for record in tqdm(annotations["images"][:100]):
    try:
        id = record['id']
        box = annotations["annotations"][id]['bbox']
        gt_box = torch.tensor([[box[0], box[1], box[0] + box[2], box[1] + box[3]]], dtype=torch.float)
        # gt_box = [int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3])]
        image = cv2.imread(image_path + record['file_name'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        results = run_example(image, task_prompt, text_input="the biggest peace of clothing.")
        bbox = results['<OPEN_VOCABULARY_DETECTION>']['bboxes'][0]
        pred_box = torch.tensor([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=torch.float)
        # pred_box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        
        iou.append(torchvision.ops.boxes.box_iou(gt_box, pred_box).item())
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #image = cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, ), 2)
        #image = cv2.rectangle(image, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 0, 255), 1)
        #cv2.imshow('test', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    except Exception as e: 
        print(id, e)
        
print("IoU: ", np.mean(iou))