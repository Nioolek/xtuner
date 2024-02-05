import argparse
import json
import random
import numpy as np
import torch
from mmdet.structures.bbox import HorizontalBoxes
from pycocotools.coco import COCO
from tqdm import tqdm


IMAGE_SIZE = 672
MIN_BOX_SIZE = 40

GLOABL_TEMPLATE = 'In images, [x, y] denotes points: top-left [0, 0], bottom-right [width-1, height-1]. Increasing x ' \
                  f'moves right; y moves down. Bounding box: [x1, y1, x2, y2]. Image size: {IMAGE_SIZE}x{IMAGE_SIZE}.'


# 生成数据


# 输入模板1：①准确bbox+开集、②准确bbox+闭集、③不准确bbox+开集、④不准确bbox+闭集
# 输入模板2：⑤大量扰动bbox+给类别，要reject、⑥大量扰动bbox+不给类别，要reject


# 准确bbox+开集
TYPE1_TEMPLATE = [
    "In the conversation below, you simply answer the category name based on what you see in the imagery inside a "
    "region <region>. "
    "The region coordinate format is [x1, y1, x2, y2], where [x1, y1] represents the top-left corner of the image, "
    f"and [x2, y2] represents the bottom-right corner coordinate. The image size is {IMAGE_SIZE}x{IMAGE_SIZE}.",
]

TYPE1_TEMPLATE_ANSWER = [
    "It's <category>.",
    "This is <category>."
]

# 准确bbox+闭集
TYPE2_TEMPLATE = [
    "In the conversation below, you simply answer the category name based on what you see in the imagery inside a "
    "region <region>. If you don't find the category name in the provided list of categories, you should output other. "
    "The region coordinate format is [x1, y1, x2, y2], where [x1, y1] represents the top-left corner of the image, "
    f"and [x2, y2] represents the bottom-right corner coordinate. The image size is {IMAGE_SIZE}x{IMAGE_SIZE}. "
    "Categories Containing {}.",
    "In the ensuing discussion, your task is to identify the category name by analyzing the imagery within a specific "
    "region <region>. If you don't find the category name in the provided list of categories, you should output other. "
    "The region coordinate format is [x1, y1, x2, y2], where [x1, y1] represents the top-left corner of the image, "
    f"and [x2, y2] represents the bottom-right corner coordinate. The image size is {IMAGE_SIZE}x{IMAGE_SIZE}. "
    "Categories will consist of {}.",
    "In the dialogue that follows, your task is to identify the category name by observing the visual elements within "
    "a specific region <region>. If you don't find the category name in the provided list of categories, you should "
    "output other. "
    "The region coordinate format is [x1, y1, x2, y2], where [x1, y1] represents the top-left corner of the image, "
    f"and [x2, y2] represents the bottom-right corner coordinate. The image size is {IMAGE_SIZE}x{IMAGE_SIZE}. "
    "Categories include {}.",
    "During the conversation that follows, your role is to determine the category name by examining the visual "
    "representation within a designated region <region>. If you don't find the category name in the provided list of "
    "categories, you should output other. "
    "The region coordinate format is [x1, y1, x2, y2], where [x1, y1] represents the top-left corner of the image, "
    f"and [x2, y2] represents the bottom-right corner coordinate. The image size is {IMAGE_SIZE}x{IMAGE_SIZE}. "
    "Categories will contain {}."
]

TYPE2_TEMPLATE_ANSWER = [
    "It's <category>.",
    "This is <category>."
]

# 不准确bbox+开集
TYPE3_TEMPLATE = [
    "In the conversation below, you simply answer the category name based on what you see in the imagery inside a "
    "region <region>. "
    "If you find that the region is inaccurate, refine the region and output region as [x1, y1, x2, y2].",
    "The region coordinate format is [x1, y1, x2, y2], where [x1, y1] represents the top-left corner of the image, "
    f"and [x2, y2] represents the bottom-right corner coordinate. The image size is {IMAGE_SIZE}x{IMAGE_SIZE}.",
]

TYPE3_TEMPLATE_ANSWER = [
    "It's <category>. The more accurate bounding boox coordinates are <region>.",
    "This is <category>. The more accurate bounding boox coordinates are <region>."
]

# 不准确bbox+闭集
TYPE4_TEMPLATE = [
    "In the conversation below, you simply answer the category name based on what you see in the imagery inside a "
    "region <region>. "
    "If you find that the region is inaccurate, refine the region and output region as [x1, y1, x2, y2].",
    "If you don't find the category name in the provided list of categories, you should output other. "
    "The region coordinate format is [x1, y1, x2, y2], where [x1, y1] represents the top-left corner of the image, "
    f"and [x2, y2] represents the bottom-right corner coordinate. The image size is {IMAGE_SIZE}x{IMAGE_SIZE}. "
    "Categories Containing {}.",
]

TYPE4_TEMPLATE_ANSWER = [
    "It's <category>. The more accurate bounding boox coordinates are <region>.",
    "This is <category>. The more accurate bounding boox coordinates are <region>."
]

# 5
REJECT_TEMPLATE_ANSWER = [
    "I can't identify the category."
]

input_path = '/data/public_datasets/coco/annotations/instances_train2017.json'

# 每个类别要多少个instance
NUM_PER_CAT = 1000

# 50%比例是reject样本
REJ_RATE = 0.5

# 正样本中，开集的比例
OPEN_RATE = 0.5

# 正样本中，准确bbox的比例
ACCURATE_RATE = 0.5

MIN_BBOX_SIZE = 40

# IOU threshold 比较当前物体与其他物体的iou，如果存在大于这个值的物体，那么就重新选择
IOU_THRESHOLD = 0.6

# perturb后与原bbox的IOU阈值
PERTURB_IOU_THRESHOLD = 0.6

# reject样本与原bbox的IOU阈值
REJECT_IOU_THRESHOLD_MAX = 0.5
REJECT_IOU_THRESHOLD_MIN = 0.1


def random_perturb(now_cat_ann, img_info):
    org_bbox = HorizontalBoxes(np.array([convert_xywh_to_xyxy(now_cat_ann['bbox'])]))

    at_num = 0
    while 1:
        at_num += 1
        bbox = org_bbox.clone()
        # 中心点随机偏移
        if random.random() < 0.9:
            h, w = bbox.heights, bbox.widths
            scale = 0.2
            x_d = w * scale * (2 * random.random() - 1)
            y_d = h * scale * (2 * random.random() - 1)
            # bbox.translate_((x_d, y_d))
            bbox.translate_(torch.Tensor([x_d, y_d]))

        # 随机缩放
        if random.random() < 0.9:
            # 获取中心点
            center = bbox.centers
            scale = 0.2
            h_scale_factor = 1. + scale * (2 * random.random() - 1)
            w_scale_factor = 1. + scale * (2 * random.random() - 1)
            bbox.translate_(-center[0])
            bbox.rescale_((w_scale_factor, h_scale_factor))
            bbox.translate_(center[0])

        # 控制在0~wh内
        bbox.clip_((img_info['height'], img_info['width']))

        # 判断与原框的iou
        iou = bbox.overlaps(org_bbox, bbox)
        # print('per ioumax', iou.max(), org_bbox)
        if iou.max() > PERTURB_IOU_THRESHOLD:
            break
        if at_num % 100 == 0:
            print('per', at_num)
    return bbox.tensor.numpy().tolist()[0]


def random_perturb_reject(now_cat_ann, img_info):
    # xywh -> xyxy
    org_bbox = HorizontalBoxes(np.array([convert_xywh_to_xyxy(now_cat_ann['bbox'])]))

    try_num = 0
    while 1:
        try_num += 1
        bbox = org_bbox.clone()
        # 中心点随机偏移
        if random.random() < 0.9:
            h, w = bbox.heights, bbox.widths
            scale = 0.3
            x_d = w * scale * (2 * random.random() - 1)
            y_d = h * scale * (2 * random.random() - 1)
            # print('!!!!!!', x_d, y_d)
            bbox.translate_(torch.Tensor([x_d, y_d]))

        # 随机缩放
        if random.random() < 0.9:
            # 获取中心点
            center = bbox.centers
            scale = 0.3
            h_scale_factor = 1. + scale * (2 * random.random() - 1)
            w_scale_factor = 1. + scale * (2 * random.random() - 1)
            # print('center', center)
            # print('00000', bbox, center)
            bbox.translate_(-center[0])
            # print('11111', bbox)
            bbox.rescale_((w_scale_factor, h_scale_factor))
            # print('22222', bbox, center)
            bbox.translate_(center[0])
            # print('33333', bbox)

        # 控制在0~wh内
        bbox.clip_((img_info['height'], img_info['width']))

        # 判断与原框的iou
        iou = bbox.overlaps(org_bbox, bbox)
        # print('detail', iou.max(), org_bbox, bbox)
        if iou.max() < REJECT_IOU_THRESHOLD_MAX and (iou.max() > REJECT_IOU_THRESHOLD_MIN):
            break
        if try_num % 100 == 0:
            print(try_num)

        # TODO: 要考虑随机扰动后的bbox与其他物体的iou
    return bbox.tensor.numpy().tolist()[0]


def judge_bbox(bbox, img_info, ann):
    # 整理bbox信息
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
    inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))

    if inter_w * inter_h == 0:
        return False

    # TODO: 把ann里area的判断往前提
    if ann['area'] <= 0 or w < 1 or h < 1:
        return False

    if ann.get('iscrowd', False):
        return False

    bbox_xyxy = [int(x1), int(y1), int(x1 + w), int(y1 + h)]

    # resize to IMAGE_SIZE
    old_w = img_info['width']
    old_h = img_info['height']
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
    bbox_xyxy = [
        int(bbox_xyxy[0] * neww / img_info['width']) + padding_w,
        int(bbox_xyxy[1] * newh / img_info['height']) + padding_h,
        int(bbox_xyxy[2] * neww / img_info['width']) + padding_w,
        int(bbox_xyxy[3] * newh / img_info['height']) + padding_h,
    ]

    # 过滤掉特别小的框
    new_h = bbox_xyxy[3] - bbox_xyxy[1]
    new_w = bbox_xyxy[2] - bbox_xyxy[0]
    if new_h < MIN_BBOX_SIZE or new_w < MIN_BBOX_SIZE:
        return False
    return bbox_xyxy


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


def calculate_bboxes_iou(box, bboxes):
    # TODO: 写法比较丑陋
    return [bbox_iou(box, b) for b in bboxes]


def convert_xywh_to_xyxy(bbox):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def sample_img(img_id, coco, now_cat_id, names):
    img_info = coco.loadImgs([img_id])[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    # print(ann)

    now_cat_flag = np.array([i['category_id'] == now_cat_id for i in anns], dtype=bool)
    # print('flag', now_cat_flag)
    # 随机选择now_cat_flag中为True的index
    now_cat_index = np.where(now_cat_flag)[0]
    # 随机从now_cat_index中选择一个
    now_cat_index = random.choice(now_cat_index)
    # print('now_cat_index', now_cat_index)
    # 当前选择的instance的标注信息
    now_cat_ann = anns[now_cat_index]

    # 筛选bbox是否合格
    bboxes = [convert_xywh_to_xyxy(ann['bbox']) for i, ann in enumerate(anns) if i != now_cat_index]
    classes = [ann['category_id'] for i, ann in enumerate(anns) if i != now_cat_index]
    # 需要计算bboxes和now_cat_ann['bbox']的iou
    if len(bboxes):
        # 计算iou
        ious = np.array(calculate_bboxes_iou(convert_xywh_to_xyxy(now_cat_ann['bbox']), bboxes))
        # 如果ious中的最大值大于threshold，那么就直接return
        if ious.max() > IOU_THRESHOLD:
            return False

    # 确认是正样本还是reject样本
    if random.random() < REJ_RATE:
        # 将bbox进行大幅度的扰动
        accurate_cat_ann = now_cat_ann
        # print('xxxxx', now_cat_ann)
        # xyxy
        move_cat_bbox = random_perturb_reject(now_cat_ann, img_info)
        if move_cat_bbox:
            pass
        else:
            # 无法找到合适的bbox，直接return
            return False
        # reject样本
        move_bbox = judge_bbox(move_cat_bbox, img_info, now_cat_ann)
        accurate_bbox = judge_bbox(convert_xywh_to_xyxy(accurate_cat_ann['bbox']), img_info, now_cat_ann)

        template_index = random.choice(range(len(TYPE1_TEMPLATE)))
        # TODO: 随机选择模板
        question = TYPE1_TEMPLATE[template_index]
        answer_index = random.choice(range(len(REJECT_TEMPLATE_ANSWER)))
        # answer = TYPE1_TEMPLATE_ANSWER[answer_index].replace('<category>', names[now_cat_id])
        answer = REJECT_TEMPLATE_ANSWER[answer_index]

        data = {
            'img_id': img_id,                   # 图片id
            'img_info': img_info,               # 图片信息
            'image': img_info['file_name'],     # 图片名称
            'category': now_cat_id,             # 当前类别
            'accurate_bbox': accurate_bbox,     # 准确的bbox
            'move_bbox': move_bbox,             # 经过扰动的bbox。如果accurate_flag为True，那么move_bbox和accurate_bbox是一样的
            'open_voc_flag': False,             # 对话是否开集。True 为开集，False 为闭集
            'accurate_flag': False,             # bbox是否准确。True 为准确，False 为经过扰动
            'rej_flag': True,                   # 是否是reject样本。True 为reject，False 为正样本
            'template_type': 5,                 # 采用的模板索引
            'conversations': [{'from': 'human', 'value': '<image>\n' + question},
                              {'from': 'gpt', 'value': answer}],
            'dataset_name': 'coco'
        }
        return data
    else:
        # 正样本
        open_voc_flag = random.random() < OPEN_RATE
        accurate_flag = random.random() < ACCURATE_RATE

        # 如果需要随机扰动，就更新now_cat_ann
        if not accurate_flag:
            move_cat_bbox = random_perturb(now_cat_ann, img_info)
            accurate_cat_ann = now_cat_ann
        else:
            move_cat_ann = accurate_cat_ann = now_cat_ann
            move_cat_bbox = move_cat_ann['bbox']
        # print('1234', move_cat_bbox)
        move_bbox = judge_bbox(move_cat_bbox, img_info, now_cat_ann)
        accurate_bbox = judge_bbox(convert_xywh_to_xyxy(accurate_cat_ann['bbox']), img_info, now_cat_ann)
        # 表明两个bbox都没问题。如果move_bbox和accurate_bbox任一为False，那么就直接return
        if move_bbox and accurate_bbox:
            pass
        else:
            return False

        if accurate_flag and open_voc_flag:
            template_type = 1
            # 使用精准bbox+开集模板
            # 从TYPE1_TEMPLATE中随机抽取一个，并获得其索引
            template_index = random.choice(range(len(TYPE1_TEMPLATE)))
            question = TYPE1_TEMPLATE[template_index]

            answer_index = random.choice(range(len(TYPE1_TEMPLATE_ANSWER)))

            answer = TYPE1_TEMPLATE_ANSWER[answer_index].replace('<category>', names[now_cat_id])
        elif accurate_flag and not open_voc_flag:
            template_type = 2
            # 使用精准bbox+闭集模板
            # 从TYPE2_TEMPLATE中随机抽取一个，并获得其索引
            template_index = random.choice(range(len(TYPE2_TEMPLATE)))
            all_cat_names = ','.join(list(names.values()))
            question = TYPE2_TEMPLATE[template_index].replace('{}', all_cat_names)
            answer_index = random.choice(range(len(TYPE2_TEMPLATE_ANSWER)))
            answer = TYPE2_TEMPLATE_ANSWER[answer_index].replace('<category>', names[now_cat_id])
        elif not accurate_flag and open_voc_flag:
            template_type = 3
            # 使用不准确bbox+开集模板
            # 从TYPE3_TEMPLATE中随机抽取一个，并获得其索引
            template_index = random.choice(range(len(TYPE3_TEMPLATE)))
            question = TYPE3_TEMPLATE[template_index]
            answer_index = random.choice(range(len(TYPE3_TEMPLATE_ANSWER)))
            answer = TYPE3_TEMPLATE_ANSWER[answer_index].replace('<category>', names[now_cat_id]).replace(
                '<region>', '[%s, %s, %s, %s]' % tuple(accurate_bbox))
        else:
            template_type = 4
            # 使用不准确bbox+闭集模板
            # 从TYPE4_TEMPLATE中随机抽取一个，并获得其索引
            template_index = random.choice(range(len(TYPE4_TEMPLATE)))
            all_cat_names = ','.join(list(names.values()))
            question = TYPE4_TEMPLATE[template_index].replace('{}', all_cat_names)
            answer_index = random.choice(range(len(TYPE4_TEMPLATE_ANSWER)))
            answer = TYPE4_TEMPLATE_ANSWER[answer_index].replace('<category>', names[now_cat_id]).replace(
                '<region>', '[%s, %s, %s, %s]' % tuple(accurate_bbox))

        # TODO: 字段中加入instance id
        data = {
            'img_id': img_id,                   # 图片id
            'img_info': img_info,               # 图片信息
            'image': img_info['file_name'],     # 图片名称
            'category': now_cat_id,             # 当前类别
            'accurate_bbox': accurate_bbox,     # 准确的bbox
            'move_bbox': move_bbox,             # 经过扰动的bbox。如果accurate_flag为True，那么move_bbox和accurate_bbox是一样的
            'open_voc_flag': open_voc_flag,     # 对话是否开集。True 为开集，False 为闭集
            'accurate_flag': accurate_flag,     # bbox是否准确。True 为准确，False 为经过扰动
            'rej_flag': False,                  # 是否是reject样本。True 为reject，False 为正样本
            'template_type': template_type,     # 采用的模板索引， TODO 要修改
            'conversations': [{'from': 'human', 'value': '<image>\n' + question},
                              {'from': 'gpt', 'value': answer}],
            'dataset_name': 'coco'
        }

        return data


def coco2rrrdataset():
    coco = COCO(input_path)
    cats = coco.loadCats(coco.getCatIds())
    names = {cat['id']: cat['name'] for cat in cats}
    all_name = list(names.values())

    out_path = 'instances_train2017_rrrvlm_ovd.json'
    # {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    # 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    # 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    # 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
    # 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    # 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
    # 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    # 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    # 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
    # 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    # 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
    print(names)

    # img_ids = coco.getImgIds()
    # print(img_ids)

    cat_len = {}
    min_len = 99999999
    min_cat = ''

    # 统计每个类别的图片数量
    for cat_id, cat_name in names.items():
        img_ids = coco.getImgIds(catIds=[cat_id])
        print(cat_name, len(img_ids))
        cat_len[cat_name] = len(img_ids)
        min_len = min(min_len, len(img_ids))
        if min_len == len(img_ids):
            min_cat = cat_name
        # break
        # print(img_ids)
        # for img_id in img_ids:
        #     ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=None)
        #     anns = coco.loadAnns(ann_ids)
        #     print(anns)
        #     break
        # break
    # cat_img_ids =
    print('每个类别数量详细情况：', cat_len)
    print('最少的类别有多少张图：', min_len, '最少的类别名称：', min_cat)

    data_list = []
    for id, name in tqdm(names.items()):
        # 从每个类里面随机抽取num_instance_pre_cat个instance
        _num_instance = 0
        _total_iter_num = 0
        ids = coco.getImgIds(catIds=[id])
        random.shuffle(ids)

        if len(ids) < NUM_PER_CAT:

            cat_data_list = []
            # 如果图片数量少，那可以重复取
            while len(cat_data_list)<NUM_PER_CAT:
                now_id = random.choice(ids)
                data = sample_img(now_id, coco, id, names)
                if data:
                    cat_data_list.append(data)
        else:
            # 如果图片数量多，那就不重复取
            cat_data_list = []
            while len(cat_data_list) < NUM_PER_CAT:
                if ids:
                    now_id = random.choice(ids)
                    # 从ids中删除now_id
                    ids.remove(now_id)
                    for i in range(5):
                        data = sample_img(now_id, coco, id, names)
                        if data:
                            cat_data_list.append(data)
                            break
                else:
                    break
        print(id, len(cat_data_list))

        data_list.extend(cat_data_list)

    # 保存json
    with open(out_path, 'w') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    coco2rrrdataset()
