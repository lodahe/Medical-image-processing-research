# -*- coding: utf-8 -*-
"""
Created on 2020-3-23
@author: LeonShangguan
"""
import cv2
import numpy as np


def do_identity(image, mask):
    return image, mask


def do_horizon_flip(image, mask, p=0.5):
    if np.random.uniform(0, 1) < p:
        image = cv2.flip(image, 1, dst=None)
        mask = cv2.flip(mask, 1, dst=None)
        return image, mask
    else:
        return image, mask


def do_vertical_flip(image, mask, p=0.5):
    if np.random.uniform(0, 1) < p:
        image = cv2.flip(image, 0, dst=None)
        mask = cv2.flip(mask, 0, dst=None)
        return image, mask
    else:
        return image, mask


def do_diagonal_flip(image, mask, p=0.5):
    if np.random.uniform(0, 1) < p:
        image = cv2.flip(image, -1, dst=None)
        mask = cv2.flip(mask, -1, dst=None)
        return image, mask
    else:
        return image, mask


def do_random_rotate(image, mask, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        angle = np.random.uniform(-1, 1)*180*magnitude

        height, width = image.shape[:2]
        cx, cy = width // 2, height // 2

        transform = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


def do_CLAHE(image, mask, clipLimit=2.0, tileGridSize=(8,8), p=0.5):
    if np.random.uniform(0, 1) < p:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        gryimg = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gryimg_planes = cv2.split(gryimg)
        gryimg_planes[0] = clahe.apply(gryimg_planes[0])
        gryimg = cv2.merge(gryimg_planes)
        image = cv2.cvtColor(gryimg, cv2.COLOR_LAB2BGR)

        grymsk = cv2.cvtColor(mask, cv2.COLOR_BGR2LAB)
        grymsk_planes = cv2.split(grymsk)
        grymsk_planes[0] = clahe.apply(grymsk_planes[0])
        grymsk = cv2.merge(grymsk_planes)
        mask = cv2.cvtColor(grymsk, cv2.COLOR_LAB2BGR)
        return image, mask  # Random CLAHE Contrast Limited Adaptive Histogram Equalization
    else:
        return image, mask