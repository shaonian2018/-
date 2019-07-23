# -*- coding: utf-8 -*-

# =========================================================================================================
# @Author: YA YING
# @Date:   Created on Thu Apr 19 16:36:51 2018
# RGB to NIR all in one 
# =========================================================================================================

import sys
caffe_root='/home/nvidia/caffe' #Caffe项目路径
sys.path.append(caffe_root+'python')
import caffe
caffe.set_mode_gpu() #设置为CPU运行
from pylab import *
import cv2
import numpy as np

#deploy.prototxt文件路径
model_def = '/home/nvidia/Desktop/CY-Net3/deploy.prototxt' 
model_weights = '/home/nvidia/Desktop/CY-Net3/trained_models/yy28_iter_11400.caffemodel' #caffemodel文件的路径

net = caffe.Net(model_def,model_weights,caffe.TEST)     # use test mode (e.g., don't perform dropout)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # 通道变换，例如从(530,800,3) 变成 (3,530,800)
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

img = cv2.imread('/home/share/yy10/0118_03_01_01.png')
#testimg = cv2.resize(img, (256, 256))
#testimg = cv2.resize(img, (256, 256),interpolation=cv2.INTER_CUBIC)
#im = caffe.io.load_image(testimg)                   #加载图片
im1 = transformer.preprocess('data',img)
net.blobs['data'].data[...] = im1      #执行上面设置的图片预处理操作，并将图片载入到blob中
net.forward()
output = net.blobs['gen_image'].data[0][0]
cv2.imwrite('gen_img.png',output)
a = np.max(output)
b = np.min(output)
print a
print b

#%%
''' 根据output的max将对比度拉伸变换到[0,255] ''' 
#result = cv2.resize(output, (800, 450))
#result = cv2.resize(output,(800,450),interpolation=cv2.INTER_CUBIC)
#result = cv2.resize(output,(1024,379),interpolation=cv2.INTER_CUBIC)
#result1 = result/(-306.0)
result = output 
result1 = result/(-147.9)
#result1 += 1.6
#cv2.imshow('result',result)
#cv2.waitKey(0)
cv2.imwrite('output.png',result1)

#%%
''' 根据output的max将对比度拉伸变换到原始trainNIR的范围 '''
#nir = cv2.imread('E:/yy/st/Env/DeepVein/0118_03_01_01.png')
result2 = result/(-162.6)
#result2 += 4.1
#result2 = result/(-594.0)
cv2.imwrite('output2.png',result2)

#%% 
''' 对结果进行直方图均衡化 '''
#r3 = cv2.imread('output.png', 0) #255
r3 = cv2.imread('output2.png', 0) #原NIR
#result2 = result/(-337.0)
result3 = cv2.equalizeHist(r3) #灰度图像直方图均衡化
cv2.imwrite('output3.png',result3)

#%% 
''' 对结果进行CLAHE处理 '''
clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(10,10))
cl1 = clahe.apply(r3)
cv2.imwrite('output_clahe.png',cl1)
