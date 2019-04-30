#!/usr/bin/env python3

import os

import torch
from PIL import Image
from torchvision import transforms as T

from data import BaseDataset


class CelebaDataset(BaseDataset):
    """Dataset class for the CelebA dataset."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(crop_size=128)
        return parser

    def __init__(self, opt, random_flip=True) -> None:
        """Initialize and preprocess the CelebA dataset."""
        super().__init__(opt)
        self._root = self.root
        t = [
            T.CenterCrop(178),
            T.Resize(128),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
        # Don't randomly flip images in test mode
        if not opt.isTrain or not random_flip:
            t.pop(2)
        self._transform = T.Compose(t)
        self._preprocess()
        # Choose split
        self._split_idx = self._train_idx if opt.isTrain else self._test_idx
        # Filter
        target_attrs = [20, 35] # male, wearing hat
        male_w_hat = self._split_idx[(self._attrs[self._split_idx][:, target_attrs] == torch.tensor([True, True])).all(-1)]
        male_no_hat = self._split_idx[(self._attrs[self._split_idx][:, target_attrs] == torch.tensor([True, False])).all(-1)]
        self._attr_splits = [male_no_hat, male_w_hat]

    def _preprocess(self):
        self._image_ids = []
        attrs = []
        with open(os.path.join(self._root, 'Anno', 'list_attr_celeba.txt'), 'r') as f:
            # skip first 2 lines
            f.readline()
            f.readline()
            for line in f:
                line = line.split()
                self._image_ids.append(line.pop(0))
                attrs.append(list(map(int, line)))
        attrs = torch.tensor(attrs)
        # convert to binary
        self._attrs = (attrs == 1)
        # 0 - train, 1 - val, 2 - test
        splits = [[], [], []]
        with open(os.path.join(self._root, 'Eval', 'list_eval_partition.txt')) as f:
            for i, line in enumerate(f):
                image_id, split_id = line.split()
                assert image_id == self._image_ids[i], 'image ID mismatch for ' + image_id
                splits[int(split_id)].append(i)
        self._train_idx, self._val_idx, self._test_idx = map(torch.tensor, splits)

    def __getitem__(self, index):
        # male w/o hat
        subset = self._attr_splits[0]
        a = self._image_ids[subset[index % len(subset)]]
        # male w/ hat
        subset = self._attr_splits[1]
        b = self._image_ids[subset[index % len(subset)]]

        image_ids = [a, b]
        images = [Image.open(os.path.join(self._root, 'Img', 'img_align_celeba', im)) for im in image_ids]
        if self._transform is not None:
            images = list(map(self._transform, images))
        return {'A': images[0], 'B': images[1]}

    def __len__(self):
        return max(map(len, self._attr_splits))
