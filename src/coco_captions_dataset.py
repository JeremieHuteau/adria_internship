import os
import json
from collections import defaultdict

import cv2
import torch
from PIL import Image

def load_annotations_file(path):
    with open(path, 'r', encoding='utf8') as f:
        annotations = json.load(f)
    return annotations

def identity(x): return x

class CocoCaptions(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_file, 
            transform=None,
            #image_transform=None, caption_transform=None,
            image_select=None, caption_select=None):
        super(CocoCaptions, self).__init__()

        self.images_dir = images_dir
        self.annotations_file = annotations_file

        #self.image_transform = image_transform
        #self.caption_transform = caption_transform
        self.transform = transform

        if image_select is None:
            image_select = identity
        self.image_select = image_select
        if caption_select is None:
            caption_select = identity
        self.caption_select = caption_select

        self.data = load_annotations_file(annotations_file)

        self.images = {}
        self.img2ann = defaultdict(list)

        self._create_index()
        self.img_ids = list(sorted(self.images.keys()))

    def _create_index(self):
        for ann in self.data['annotations']:
            self.img2ann[ann['image_id']].append(ann)
        for img in self.data['images']:
            self.images[img['id']] = img

    def __getitem__(self, idx):
        image_id = self.img_ids[idx]

        images_data = [self.images[image_id]]
        images_data = self.image_select(images_data)

        images = []
        for image_data in images_data:
            image = cv2.imread(os.path.join(
                self.images_dir, image_data['file_name'])) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        image_ratios = [image.shape[0]/image.shape[1] for image in images]

        annotations_data = self.img2ann[image_id]
        annotations_data = self.caption_select(annotations_data)

        captions = [ann_data['caption'] for ann_data in annotations_data]

        if self.transform is not None:
            data = self.transform(image=images[0], text=captions[0])
            images = [data['image']]
            captions = [data['text']]

        #if self.image_transform is not None:
        #    images = [self.image_transform(image) for image in images]
        #if self.caption_transform is not None:
        #    captions = [self.caption_transform(caption) for caption in captions]

        return {
            'images': images,
            'captions': captions,

            'image_ratios': image_ratios,

            'images_data': images_data,
            'annotations_data': annotations_data,
        }


    def __len__(self):
        return len(self.img_ids)

def captions(annotations_file):
    annotations = load_annotations_file(annotations_file)['annotations']
    return map(lambda ann: ann['caption'], annotations)

