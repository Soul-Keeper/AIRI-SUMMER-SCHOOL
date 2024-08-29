import json
import torch

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import ViltProcessor, ViltForQuestionAnswering

image_path = "../data/vto+color/train/"
annotations_path = "../data/vto+color/train/_annotations.coco.json"

f = open(annotations_path)
annotations = json.load(f)

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to('cuda')
print(f"Model size: {model.num_parameters()}")

question = 'What is the biggest piece of clothing on the picrure, tshirt or sweatshirt?'
answers = []
gt_cls = []

for record in tqdm(annotations["images"][:1000]):
    try:
        ground_truth_cls = record['extra']['user_tags'][0].split('-')[1]
        image = Image.open(image_path + record['file_name']).convert('RGB')
        encoding = processor(image, question, return_tensors="pt").to('cuda')

        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        
        answers.append(model.config.id2label[idx])
        gt_cls.append(ground_truth_cls)
        
        print(gt_cls[-1], answers[-1])

    except Exception as e: 
        print(id, e)

print("cls acc: ", accuracy_score(gt_cls, answers))

