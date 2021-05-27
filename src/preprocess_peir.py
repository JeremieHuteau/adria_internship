import argparse
import os
import json
from collections import defaultdict

import image_preprocessing

images_dir = 'images/'
annotations_dir = 'annotations/'

def main(args):
    annotation_data = json.load(open(annotations_file, 'r'))

    annotation_data = select_PEIR_category(annotation_data, 'Histology')
    annotation_data = group_images(annotation_data)

    preprocessed_annotations_file = os.path.join(
            args.preprocessed_dir, annotations_dir, 
            os.path.basename(args.annotations_file))
    json.dump(annotation_data, open(preprocessed_annotations_file, 'w'), 
            indent=2)

    image_paths = map(
            lambda img: os.path.join(
                args.raw_dir, images_dir, img['filename']),
            annotation_data['images'])
    target_size = (256, 256)
    image_preprocessing.resize_images(image_paths, target_size, 
            os.path.join(args.preprocessed_dir, images_dir))

def select_PEIR_category(annotation_data, category):
    categories = {
        cat['id']: cat
        for cat in annotation_data['categories']
    }
    histology_categories = {
        cat['id']: cat
        for cat in annotation_data['categories']
        if cat['name'] == category
    }

    histology_images = {
        img['id']: img
        for img in annotation_data['images']
        if any((cat in histology_categories for cat in img['categories']))
    }

    histology_annotations = {
        ann['id']: ann
        for ann in annotation_data['annotations']
        if ann['image_id'] in histology_images
    }

    outliers = list(map(
        lambda ann: (ann, histology_images[ann['image_id']]),
        filter(
            lambda ann: not ann['caption'].startswith(category.upper()),
            histology_annotations.values())))
    for ann, img in outliers:
        histology_images.pop(img['id'])
        histology_annotations.pop(ann['id'])

    clean_data = {
        'categories': categories.values(),
        'images': histology_images.values(),
        'annotations': histology_annotations.values(),
    }
    return clean_data

def group_images(annotation_data):
    def annotation_key_fn(annotation):
        return annotation['caption']

    annkey2anns = defaultdict(list)
    for ann in annotation_data['annotations']:
        annotations[annotation_key_fn(ann)].append(ann)

    images = [

    ]

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #parser.add_argument('dataset', type=str, choices=['PEIR'])
    parser.add_argument('raw_dir', type=str)
    parser.add_argument('annotations_file', type=str)
    parser.add_argument('preprocessed_dir', type=str)

    args = parser.parse_args()
    main(args)

