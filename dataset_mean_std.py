import os 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import tqdm

img_dir = '/home/sommarjobbare/images_books'

R_channel_sum = 0
G_channel_sum = 0
B_channel_sum = 0
num = 0

for file_name in tqdm.tqdm(os.listdir(img_dir)):
    img = imread(os.path.join(img_dir,file_name))
    R_channel_sum = R_channel_sum + np.sum(img[:,:,0])
    G_channel_sum = G_channel_sum + np.sum(img[:,:,1])
    B_channel_sum = B_channel_sum + np.sum(img[:,:,2])

    num = num + (img.shape[0]*img.shape[1])

R_mean = R_channel_sum/num
G_mean = G_channel_sum/num
B_mean = B_channel_sum/num

R_sqdiff = 0
G_sqdiff = 0
B_sqdiff = 0

for file_name in tqdm.tqdm(os.listdir(img_dir)):
    img = imread(os.path.join(img_dir,file_name))
    R_sqdiff = R_sqdiff + np.sum((img[:, :, 0] - R_mean) ** 2)
    G_sqdiff = G_sqdiff + np.sum((img[:, :, 1] - G_mean) ** 2)
    B_sqdiff = B_sqdiff + np.sum((img[:, :, 2] - B_mean) ** 2)

R_std = np.sqrt(R_sqdiff / num)
G_std = np.sqrt(G_sqdiff / num)
B_std = np.sqrt(B_sqdiff / num)

print('R_mean:',R_mean)
print('G_mean:',G_mean)
print('B_mean:',B_mean)
print('R_mean_norm:',R_mean/255)
print('G_mean_norm:',G_mean/255)
print('B_mean_norm:',B_mean/255)
print('R_std:',R_std)
print('G_std:',G_std)
print('B_std:',B_std)
print('R_std_norm:',R_std/255)
print('G_std_norm:',G_std/255)
print('B_std_norm:',B_std/255)


