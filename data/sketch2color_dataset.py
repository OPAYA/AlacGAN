from torch.utils.data import Dataset
import os
from os import listdir
from PIL import Image
from torchvision import transforms
import cv2
from albumentations import (
    HorizontalFlip, RandomResizedCrop, Resize, OneOf, Compose, RandomBrightnessContrast, HueSaturationValue
)
from albumentations.pytorch.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.transforms import Scale, CenterCrop

import random
import math
import numpy as np

class RandomSizedCrop(object):

    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.9, 1.) * area
            aspect_ratio = random.uniform(7. / 8, 8. / 7)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))

def get_train_transform(input_height, input_width):
    return Compose([
        RandomBrightnessContrast(),
        HueSaturationValue()
    ]),\
    Compose([
        OneOf([
            RandomResizedCrop(input_height, input_width, interpolation=cv2.INTER_NEAREST),
            RandomResizedCrop(input_height, input_width, interpolation=cv2.INTER_LINEAR),
            RandomResizedCrop(input_height, input_width, interpolation=cv2.INTER_CUBIC),
            RandomResizedCrop(input_height, input_width, interpolation=cv2.INTER_AREA),
            RandomResizedCrop(input_height, input_width, interpolation=cv2.INTER_LANCZOS4),
            Resize(input_height, input_width)
        ], p=1.0),
        HorizontalFlip(),
        ToTensor()
    ])

VTrans = transforms.Compose([
    RandomSizedCrop(256 // 4, Image.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_test_transform(input_height, input_width):
    return Compose([
        Resize(input_height, input_width),
        ToTensor()
    ])

class Sketch2ColorDataset(Dataset):
    def __init__(self, dataset, phase, input_height, input_width, processed_dir):
        super().__init__()
        self.phase = phase

        assert phase in ['train', 'val', 'test']
        
        sketch_dir = os.path.join(processed_dir, dataset, phase, 'sketch')
        color_dir = os.path.join(processed_dir, dataset, phase, 'color')
        
        sketch_fnames = sorted(listdir(sketch_dir), key=lambda x:int(x.split('.')[0]))
        color_fnames = sorted(listdir(color_dir), key=lambda x:int(x.split('.')[0]))

        self.sketch_fnames = [os.path.join(sketch_dir, fname) for fname in sketch_fnames]
        self.color_fnames = [os.path.join(color_dir, fname) for fname in color_fnames]

        if phase == 'train':
            transform = get_train_transform(input_height, input_width)
        else:
            transform = get_test_transform(input_height, input_width)
        self.transform = Compose(transform, additional_targets={'image2': 'image'})

    def __getitem__(self, idx):
        """
        output:
        sketch image: B x 1 x H x W image tensor scaled to [0, 1]
        color image: B x 3 x H x W image tensor scaled to [0, 1]
        """
        # sketch_img = cv2.cvtColor(cv2.imread(self.sketch_fnames[idx]), cv2.COLOR_BGR2RGB)

        # color_img = cv2.cvtColor(cv2.imread(self.color_fnames[idx]), cv2.COLOR_BGR2RGB)

        sketch_img = np.array(Image.open(self.sketch_fnames[idx]).convert('RGB'))
        color_img = np.array(Image.open(self.color_fnames[idx]).convert('RGB'))
        color_img2 = Image.open(self.color_fnames[idx]).convert('RGB')

        if self.phase == 'train':
            out3 = VTrans(color_img2)
            transform1, transform2 = self.transform
            color_img = transform1(image=color_img)['image']
            aug_output = transform2(image=sketch_img, image2=color_img)
            out1, out2 = aug_output['image'], aug_output['image2']
        else:
            out3 = VTrans(color_img2)
            aug_output = self.transform(image=sketch_img, image2=color_img)
            out1, out2 = aug_output['image'], aug_output['image2']
        
        # RGB to GRAY
        out1 = 0.299 * out1[0:1,:,:] + 0.587 * out1[1:2,:,:] + 0.114 * out1[2:3,:,:]

        # color, mask, gray
        return out2, out3, out1

    def __len__(self):
        return len(self.sketch_fnames)

