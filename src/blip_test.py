import json
import torch

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import BlipProcessor, BlipForQuestionAnswering

image_path = "../data/vto+color/train/"
annotations_path = "../data/vto+color/train/_annotations.coco.json"

f = open(annotations_path)
annotations = json.load(f)

processor = BlipProcessor.from_pretrained("ybelkada/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("ybelkada/blip-vqa-capfilt-large", torch_dtype=torch.float16).to("cuda")
print(f"Model size: {model.num_parameters()}")

question = 'What is person working with, sweatshirt or tshirt?'
answers = []
gt_cls = []

for record in tqdm(annotations["images"][:1000]):
    try:
        ground_truth_cls = record['extra']['user_tags'][0].split('-')[1]
        image = Image.open(image_path + record['file_name']).convert('RGB')
        inputs = processor(image, question, return_tensors="pt").to("cuda", torch.float16)
        out = model.generate(**inputs)
        
        answers.append(processor.decode(out[0], skip_special_tokens=True))
        gt_cls.append(ground_truth_cls)
        
    except Exception as e: 
        print(id, e)

print("cls acc: ", accuracy_score(gt_cls, answers))