import os
import itertools
import operator
import random
from typing import Optional

import numpy as np
import torch
import torchvision
import albumentations as alb
import pytorch_lightning as pl

import coco_captions_dataset
import factories

def worker_init_fn(worker_id):
    """https://github.com/pytorch/pytorch/issues/5059"""
    process_seed = torch.initial_seed()
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([worker_id, base_seed])
    np.random.seed(ss.generate_state(4))

def make_transform(transform_cfgs, factory):
    individual_transforms = []
    for transform_cfg in transform_cfgs:
        individual_transforms.append(factory.create(
                transform_cfg['name'], 
                **transform_cfg['kwargs']))
    return alb.Compose(individual_transforms)

class CocoCaptionsDataModule(pl.LightningDataModule):
    images_dir = 'images/'
    annotations_dir = 'annotations/'
    train_annotations_file = annotations_dir + 'captions_train2017.json'
    val_annotations_file = annotations_dir + 'captions_val2017.json'
    test_annotations_file = annotations_dir + 'captions_test2017.json'

    def __init__(self, 
            root_dir: str, 
            batch_size: int, 
            transforms_cfg: dict,
            single_caption: bool = True,
            num_workers: int = 0, pin_memory: bool = False,
            ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_transform = make_transform(
                transforms_cfg['train'],
                factories.TransformFactory)
        self.test_transform = make_transform(
                transforms_cfg['test'],
                factories.TransformFactory)

        if single_caption:
            self.text_train_select = lambda x: [random.choice(x)]
            self.text_test_select = lambda x: [x[0]]
        else:
            raise ValueError("Multiple captions not supported due to"
                    " augmentation pipeline.")
            self.text_train_select = self.text_test_select = lambda x: x

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train = coco_captions_dataset.CocoCaptions(
                    os.path.join(self.root_dir, self.images_dir), 
                    os.path.join(self.root_dir, self.train_annotations_file),
                    transform = self.train_transform,
                    caption_select = self.text_train_select)

            self.val = coco_captions_dataset.CocoCaptions(
                    os.path.join(self.root_dir, self.images_dir), 
                    os.path.join(self.root_dir, self.val_annotations_file),
                    transform = self.test_transform,
                    caption_select = self.text_test_select)

        if stage == 'test' or stage is None:
            self.test = coco_captions_dataset.CocoCaptions(
                    os.path.join(self.root_dir, self.images_dir), 
                    os.path.join(self.root_dir, self.test_annotations_file),
                    transform = self.test_transform,
                    caption_select = self.text_test_select)

    def _collate_fn(self, batch):
        """
            batch: list of dicts. Each dict contains a list of images and a list
            of captions, for which there is metadata.

            Returns a dict, with a tensor of images, a tensor of captions, 
            tensors for the metadata of images and captions, and a tensor
            for the edgelist of images-captions indices pairs. The captions are
            sorted by length.
        """
        image_key = 'images'
        image_data_keys = ['image_ratios', 'images_data']
        def dict_to_image_tuples(d):
            images = d[image_key]
            data = [d[key] for key in image_data_keys]
            return zip(images, *data)

        image_tuples = list(itertools.chain.from_iterable(
                map(dict_to_image_tuples, batch)))
        image_lists = list(zip(*image_tuples))

        images = torch.stack(image_lists[0], dim=0)
        images_data = {
                image_data_keys[i]: value
                for i, value in enumerate(image_lists[1:])}

        sequence_key = 'captions'
        sequence_data_keys = ['annotations_data']
        def dict_to_sentence_tuples(d):
            sequences = d[sequence_key]
            lengths = map(len, sequences)
            data = [d[key] for key in sequence_data_keys]
            return zip(lengths, sequences, *data)
        sequence_tuples = list(itertools.chain.from_iterable(
                map(dict_to_sentence_tuples, batch)))

        sort_indices, sequence_tuples = zip(*sorted(
                enumerate(sequence_tuples),
                key=lambda t: t[1][0], reverse=True))
        sequence_lists = list(zip(*sequence_tuples))

        lengths, sequences = sequence_lists[:2]
        lengths = torch.tensor(lengths, dtype=torch.long)
        sequences_data = {
                sequence_data_keys[i]: value
                for i, value in enumerate(sequence_lists[2:])}

        inverse_sort_indices = [None] * len(sort_indices)
        for i, sort_index in enumerate(sort_indices):
            inverse_sort_indices[sort_index] = i

        padded_sequences = torch.zeros(lengths.numel(), lengths[0],
                dtype=torch.long)
        for i, sequence in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = sequence

        firsts = map(lambda d: d[image_key], batch)
        seconds = map(lambda d: d[sequence_key], batch)
        first_lengths = list(map(len, firsts))
        second_lengths = list(map(len, seconds))

        def accumulate(iterable, operator = operator.add):
            accumulator = 0
            for element in iterable:
                yield accumulator
                accumulator = operator(accumulator, element)

        first_starts = accumulate(first_lengths)
        second_starts = accumulate(second_lengths)

        positive_pairs = torch.tensor([
            [source, inverse_sort_indices[target]]
            for source_start, source_length, target_start, target_length 
                in zip(first_starts, first_lengths, second_starts, second_lengths)
            for source in range(source_start, source_start + source_length)
            for target in range(target_start, target_start + target_length)],
        dtype=torch.long)

        batch = {
            'images': images,
            'texts': padded_sequences,
            'lengths': lengths,
            'positive_pairs': positive_pairs,
            **images_data,
            **sequences_data,
        }
        return batch

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train,
                batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers, pin_memory=self.pin_memory,
                collate_fn=self._collate_fn,
                drop_last=True,
                worker_init_fn=worker_init_fn)
    def _shared_test_dataloader(self, dataset):
        return torch.utils.data.DataLoader(dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers, pin_memory=self.pin_memory,
                collate_fn=self._collate_fn,
                worker_init_fn=worker_init_fn)
    def val_dataloader(self):
        return self._shared_test_dataloader(self.val)
    def test_dataloader(self):
        return self._shared_test_dataloader(self.test)

    def visualize_batch(self, batch, normalization_parameters, vocabulary, num_display=2):
        import matplotlib.pyplot as plt
        import torchvision

        images = batch['images'][:num_display]
        positive_pairs = batch['positive_pairs']
        captions = batch['texts']
        #captions = list(map(lambda d: d['caption'], batch['annotations_data']))

        print(positive_pairs)
        print(captions)
        raw_captions = list(map(lambda d: d['caption'], batch['annotations_data']))
        print(*enumerate(raw_captions), sep='\n')

        image_std = torch.tensor(normalization_parameters['std']).view(-1, 1, 1)
        image_mean = torch.tensor(normalization_parameters['mean']).view(-1, 1, 1)

        images_captions = []
        for i in range(num_display):
            images_captions.append([])
        for pair in positive_pairs:
            if pair[0] < num_display:
                caption = captions[pair[1]]
                images_captions[pair[0]].append(caption)

        for i in range(images.size(0)):
            image = images[i]
            image = image * image_std + image_mean
            image = torchvision.transforms.ToPILImage()(image)

            texts = map(lambda t: t.tolist(), images_captions[i])
            texts = map(lambda t: vocabulary.idx2doc(t), texts)
            texts = map(lambda t: ' '.join(t), texts)
            texts = '\n'.join(texts)


            fig, axes = plt.subplots()
            axes.set(xlabel = texts)
            plt.imshow(image)
        plt.show()

