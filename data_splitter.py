"""
Created on 2020-3-1
@author: Yuxuan He
"""

import cv2
import os

path = 'C:/Users/15853/Desktop/IVUS_Research/DoyleyResearch-master/Code/Carotid-Data/images/'
file_path = 'C:/Users/15853/Desktop/IVUS_Research/DoyleyResearch-master/Code/Carotid-Data/images/'
mask_path = 'C:/Users/15853/Desktop/IVUS_Research/DoyleyResearch-master/Code/Carotid-Data/masks/'
# save_path = '/Users/yuxuanhe/Desktop/Research IVUS/Carotid-Data/dataset/'

a = sorted(os.listdir(path))

b = sorted(os.listdir(file_path))

c = sorted(os.listdir(mask_path))

for i in range(0, 31):
    file_name = b[i]
    print(file_name)

for i in range(0, 31):
    file_name = b[i]
    for data_file in sorted(os.listdir(path + file_name + '/')):
        img = cv2.imread('C:/Users/15853/Desktop/IVUS_Research/DoyleyResearch-master/Code/Carotid-Data/images/' + file_name + '/' + data_file)
        cv2.imwrite('C:/Users/15853/Desktop/IVUS_Research/DoyleyResearch-master/Code/Carotid-Data/image_set/' + file_name + '_' + data_file[0:4] + '.jpg',
                    img)

for i in range(0, 15):
    file_name = c[i]
    for data_file in sorted(os.listdir(path + file_name + '/')):
        img = cv2.imread('C:/Users/15853/Desktop/IVUS_Research/DoyleyResearch-master/Code/Carotid-Data/masks/' + file_name + '/' + data_file)
        cv2.imwrite('C:/Users/15853/Desktop/IVUS_Research/DoyleyResearch-master/Code/Carotid-Data/mask_set/' + file_name + '_' + data_file[0:4] + '.jpg',
            img)

