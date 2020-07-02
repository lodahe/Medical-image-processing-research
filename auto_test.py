# -*- coding: utf-8 -*-
"""
Created on 2019-3-18
@author: LeonShangguan
"""
import cv2
import glob
import numpy as np
from keras.models import load_model


def auto_test(read_path, model_name, save_path='', output_size=(420, 420)):
    """
    auto test for trained model
    :param read_path: test_image folder, type:string
    :param model_name: test_model name, type:string
    :param save_path: test_image save folder, type: string, default: same to test_image folder
    :param output_size: output image size, default: 420*420
    :return: None
    """
    model = load_model(model_name, compile=False)

    if save_path == '':
        save_path = read_path

    for filename in glob.glob(read_path + '/*.png'):
        img = cv2.imread(filename)/255
        img = cv2.resize(img, (512, 512))
        img = np.reshape(img, (1,)+img.shape)
        result = model.predict(img)[0, :, :, :]
        result = cv2.resize(result, output_size)*255
        cv2.imwrite(save_path + filename.split('/')[-1][:-4] + '_newsplit.jpg', result)


# only for my test
if __name__ == '__main__':
    auto_test(read_path='../test',
              model_name='Unet_test_allimage_clahe_10epoch.hdf5',
              save_path='../',
              output_size=(1000, 1000))
