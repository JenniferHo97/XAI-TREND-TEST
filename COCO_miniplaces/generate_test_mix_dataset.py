# -*- coding: UTF-8 -*-

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def save_img(img, path):
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0)


def generate_mix_img(coco_path, category_idx, places_path, places_file_list, num_data, target_fg_path, target_bg_path, target_mix_path):
    # get file index
    coco = COCO(coco_path)
    fg_ids = coco.catToImgs[category_idx][:num_data]

    # load coco file
    for i, fg_img_id in enumerate(tqdm(fg_ids)):
        fg_img_path = coco.loadImgs(fg_img_id)[0]['file_name']
        origin_fg_img = Image.open(os.path.join(
            root, fg_img_path)).convert('RGB')
        np_origin_fg_img = np.asarray(origin_fg_img)

        # get fg img
        ann_ids = coco.getAnnIds(imgIds=fg_img_id)
        target = coco.loadAnns(ann_ids)
        fg = 0
        mask_list = []
        for j, ann in enumerate(target):
            if ann['category_id'] == category_idx:
                mask = coco.annToMask(ann)
                mask_list.append(mask)
                fg = fg + mask

        fg_mask = np.repeat(fg, 3).reshape(
            np_origin_fg_img.shape[0], np_origin_fg_img.shape[1], 3)
        np_fg_img = fg_mask * np_origin_fg_img

        # get bg img
        origin_bg_img = Image.open(os.path.join(
            places_path, places_file_list[i % len(places_file_list)]))
        origin_bg_img = origin_bg_img.resize(
            (origin_fg_img.width, origin_fg_img.height), Image.ANTIALIAS)
        np_origin_bg_img = np.asarray(origin_bg_img)

        bg_mask = np.where(np_fg_img != 0, 0, 1)
        masked_np_bg_img = np_origin_bg_img * bg_mask
        np_mix_img = np_fg_img + masked_np_bg_img

        save_img(np_fg_img, os.path.join(target_fg_path, '{}.jpg'.format(i)))
        save_img(np_origin_bg_img, os.path.join(
            target_bg_path, '{}.jpg'.format(i)))
        save_img(np_mix_img, os.path.join(target_mix_path, '{}.jpg'.format(i)))


if __name__ == '__main__':
    # origin data path of COCO
    root = './COCO2017/train2017'
    annFile_train = './COCO2017/annotations/instances_train2017.json'
    annFile_val = './COCO2017/annotations/instances_val2017.json'

    # origin data path of miniplaces
    bedroom_path = './miniplaces/images/train/b/bedroom'
    coast_path = './miniplaces/images/train/c/coast'

    # target data path
    fg_cat_path = './data/fg_imgs/cat/'
    fg_airplane_path = './data/fg_imgs/airplane/'
    bg_bedroom_path = './data/bg_imgs/bedroom'
    bg_coast_path = './data/bg_imgs/coast'
    cat_bedroom_path = './data/mix_imgs/cat_bedroom/'
    cat_coast_path = './data/test_mix_imgs/cat_coast/'
    airplane_bedroom_path = './data/test_mix_imgs/airplane_bedroom/'
    airplane_coast_path = './data/mix_imgs/airplane_coast/'
    bedroom_file_list = os.listdir(bedroom_path)
    coast_file_list = os.listdir(coast_path)

    # target category of COCO
    cat_idx = 17
    airplane_idx = 5

    # generate dataset
    num_data = 1000
    generate_mix_img(annFile_train, cat_idx,
                     coast_path, coast_file_list, num_data, fg_cat_path, bg_coast_path, cat_coast_path)
    generate_mix_img(annFile_train, airplane_idx,
                     bedroom_path, bedroom_file_list, num_data, fg_airplane_path, bg_bedroom_path, airplane_bedroom_path)
