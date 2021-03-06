import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.arcface_models import ResNet

netArc_checkpoint = torch.load('models/arcface_checkpoint.tar', map_location=torch.device('cuda'))
netArc = netArc_checkpoint['model'].module

def get_id(attr_img_align_crop_pil):
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    attr_img = transformer(attr_img_align_crop_pil).unsqueeze(0)
    attr_img = attr_img.to('cuda')
    attr_img_arc = F.interpolate(attr_img,size=(112,112), mode='bicubic')
    attr_id = netArc(attr_img_arc)
    attr_id = F.normalize(attr_id, p=2, dim=1)

    # print(attr_id.shape)
    return attr_id.tolist()
