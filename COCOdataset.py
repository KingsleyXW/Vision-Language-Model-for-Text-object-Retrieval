import os
# import pandas as pd
import torch
# from torchvision.io import read_image
from torch.utils.data import Dataset
# import json

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
# BICUBIC = InterpolationMode.BICUBIC

import sys 
sys.path.append(os.path.dirname(__file__))

from simple_tokenizer import SimpleTokenizer as Tokenizer
from typing import Any, Union, List
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

BICUBIC = InterpolationMode.BICUBIC
tokenizer = Tokenizer()

class COCODataset(Dataset):
    def __init__(self, bbox_file, cap_file, image_root, image_transform=None, text_transform=None):
        self.bbox_ann = COCO(bbox_file)
        self.caption_ann = COCO(cap_file)
        self.imgID = list(self.bbox_ann.imgs.keys())
        self.image_root = image_root
        cats = self.bbox_ann.loadCats(self.bbox_ann.getCatIds())
        self.cats = [cat['name'] for cat in cats]
        if image_transform:
            self.image_transform = image_transform
        else:
            self.image_transform = _transform(224)
        if image_transform:
            self.text_transform = text_transform
        else:
            self.text_transform = tokenize

    def __len__(self):
        return len(self.imgID)

    

    def __getitem__(self, index):
        img_id = self.imgID[index]

        bbox, cat = self.__getBbox(img_id)
        captions = self.__getCaptions(img_id)

        filename = self.__getImageFileName(img_id)
        image = self.__get_raw_data(filename)
        image = self.image_transform(image)
        captions = self.text_transform(captions)
        return image, captions, bbox, cat
    
    @property
    def idx_to_class(self):
        cats = self.bbox_ann.loadCats(self.bbox_ann.getCatIds())
        return {x['id']:x['name'] for x in cats} 

    def __getImageFileName(self, id):
        data = self.bbox_ann.loadImgs(id)
        return data[0]['file_name']

    def __getBbox(self, id):
        ann_ids = self.bbox_ann.getAnnIds(imgIds=id)
        bAnns = self.bbox_ann.loadAnns(ann_ids)
        bbox = torch.tensor([x['bbox'] for x in bAnns])
        cat =  torch.tensor([x['category_id'] for x in bAnns])
        return bbox, cat

    def __getCaptions(self, id):
        ann_ids = self.caption_ann.getAnnIds(imgIds=id)
        cAnns = self.caption_ann.loadAnns(ann_ids)
        captions = [x['caption'] for x in cAnns]
        return captions

    def getCatName(self, id):
        # id is coco bbox id, not the image index in this dataset!
        return self.bbox_ann.loadCats(id)[0]['name']
    
    def __get_raw_data(self, filename):
        img_path = os.path.join(self.image_root, filename)
        image = Image.open(img_path) #0-1
        return image

    def get_raw_data(self, index):
        img_id = self.imgID[index]
        filename = self.__getImageFileName(img_id)
        return self.__get_raw_data(filename)

    def showBboxAnn(self, index):
        img_id = self.imgID[index]
        img = self.bbox_ann.loadImgs(img_id)[0]
        I = self.__get_raw_data(img['file_name'])
        plt.imshow(I); plt.axis('off')
        annIds = self.bbox_ann.getAnnIds(imgIds=img['id'])
        anns = self.bbox_ann.loadAnns(annIds)
        self.bbox_ann.showAnns(anns, draw_bbox=True)

    def showCaptionAnn(self, index):
        img_id = self.imgID[index]
        img = self.caption_ann.loadImgs(img_id)[0]
        annIds = self.caption_ann.getAnnIds(imgIds=img['id'])
        anns = self.caption_ann.loadAnns(annIds)
        self.caption_ann.showAnns(anns)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        # Resize(n_px, interpolation=BICUBIC),
        # CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
