import json
import os.path

import cv2
from PIL import Image
import torchvision.transforms.functional as F

# from .coco_data1 import bbox_iou

with open('instances_train2017_rrrvlm_ovd.json', 'r') as file:
    data_list = json.load(file)

def bbox_iou(box1, box2):
    # Calculate the (x1, y1, x2, y2) coordinates of the intersection of box1 and box2.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Calculate the Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou


# 随机选择一例
import random
# data = random.choice(data_list)
data = data_list[0]
# data = {
#             'img_id': img_id,                   # 图片id
#             'img_info': img_info,               # 图片信息
#             'image': img_info['file_name'],     # 图片名称
#             'category': now_cat_id,             # 当前类别
#             'accurate_bbox': accurate_bbox,     # 准确的bbox
#             'move_bbox': move_bbox,             # 经过扰动的bbox。如果accurate_flag为True，那么move_bbox和accurate_bbox是一样的
#             'open_voc_flag': False,             # 对话是否开集。True 为开集，False 为闭集
#             'accurate_flag': False,             # bbox是否准确。True 为准确，False 为经过扰动
#             'rej_flag': True,                   # 是否是reject样本。True 为reject，False 为正样本
#             'template_type': 5,                 # 采用的模板索引
#             'conversations': [{'from': 'human', 'value': '<image>\n' + question},
#                               {'from': 'gpt', 'value': answer}],
#             'dataset_name': 'coco'
#         }

img_path = os.path.join('/data/public_datasets/coco/train2017', data['image'])
img = cv2.imread(img_path)

img_info = data['img_info']
old_w = img_info['width']
old_h = img_info['height']
IMAGE_SIZE = 672
scale_factor = min(IMAGE_SIZE / max(old_h, old_w),
                       IMAGE_SIZE / min(old_h, old_w))
neww = int(old_w * float(scale_factor) + 0.5)
newh = int(old_h * float(scale_factor) + 0.5)

if neww > newh:
    padding_h = (neww - newh) // 2
    padding_w = 0
else:
    padding_w = (newh - neww) // 2
    padding_h = 0
import numpy as np
new_image = np.zeros((672, 672, 3), dtype=np.uint8)
new_image[padding_h:padding_h+newh, padding_w:padding_w+neww] = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_CUBIC)




# img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_CUBIC)


# image = Image.open(img_path).convert('RGB')
# old_w, old_h = F.get_image_size(image)
# scale_factor = min(672 / max(old_h, old_w),
#                    672 / min(old_h, old_w))
# neww = int(old_w * float(scale_factor) + 0.5)
# newh = int(old_h * float(scale_factor) + 0.5)
# image = F.resize(image, size=(newh, neww), interpolation=F.InterpolationMode.BICUBIC)
#
#
#
# if True
#     image = expand2square(
#         image,
#         tuple(
#             int(x * 255) for x in self.image_processor.image_mean))


accurate_bbox = data['accurate_bbox']
move_bbox = data['move_bbox']
# 画出准确的bbox
cv2.rectangle(new_image, (accurate_bbox[0], accurate_bbox[1]), (accurate_bbox[2], accurate_bbox[3]), (0, 255, 0), 2)
# # 画出扰动的bbox
cv2.rectangle(new_image, (move_bbox[0], move_bbox[1]), (move_bbox[2], move_bbox[3]), (0, 0, 255), 2)
# cv2.imshow('1', img)
cv2.imwrite('1.jpg', new_image)
print(data)

print(bbox_iou(accurate_bbox, move_bbox))


