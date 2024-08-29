import cv2
import json
import torch
import numpy as np
import torchvision
import torchvision.transforms as T

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.functional import InterpolationMode

image_path = "../data/vto+color/train/"
annotations_path = "../data/vto+color/train/_annotations.coco.json"

f = open(annotations_path)
annotations = json.load(f)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = 'OpenGVLab/InternVL2-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    use_flash_attn=False,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


print(f"Model size: {model.num_parameters()}")

iou = []
gt_cls = []
pred_cls = []


for record in tqdm(annotations["images"][:1]):
    try:
        id = record['id']
        box = annotations["annotations"][id]['bbox']
        #gt_box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        #t_box = torch.tensor([[box[0], box[1], box[0] + box[2], box[1] + box[3]]], dtype=torch.float)
        ground_truth_color = record['extra']['user_tags'][1].split('-')[1]
        #ground_truth_cls = record['extra']['user_tags'][0].split('-')[1]
        #image = Image.open('/home/user8/nsidelnikov/data/kroy/train/images/10_jpg.rf.568fc9f9aa33cd0799b9b18440b6a10e.jpg').convert('RGB')
        #image_cv = cv2.imread(image_path + record['file_name'])
        
        #print(ground_truth_cls)

        pixel_values = load_image(image_path + record['file_name'], max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        question = '<image>\nWhat color is piece of clothing person workin with? Choose from black, brown, nude, white and pink.'
        #question = '<image>\nProvide segmentation masks of patterns placed on the fabric.'
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        print(f'User: {question}\nAssistant: {response}')
        
        #bbox = response.split(', ')
        #bbox = [int(n.replace('.', '')) for n in bbox]
        
        gt_cls.append(ground_truth_color)
        pred_cls.append(response.lower().replace('-',''))
        
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