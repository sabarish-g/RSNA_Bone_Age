import numpy as np
import tensorflow as tf
import random, os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50

from keras import applications
path = '/shared/kgcoe-research/mil/BoneAge/Bone_Age/new_split'

#The train and test paths
train_path = os.path.join(path, 'train2')
test_path = os.path.join(path, 'val2')



#Randomly select image:
train_image_list = glob.glob(train_path+'/*')
test_image_list = glob.glob(test_path+'/*')
random_image = random.choice(train_image_list)


#Read the sample image
sample_image = cv2.imread(random_image)
# print("The shape of the image is", sample_image.shape)
# cv2.imwrite('/home/sxg8458/KDD_Project/sample_image.png',sample_image)



'''
This code is for the document to be submitted on 12th October
'''
img_width = 128
img_height = 128    

# build the VGG16 network
model = VGG16(include_top=False, weights='imagenet')
input_img = model.input
# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# print(layer_dict)

from keras import backend as K

layer_name = 'block5_conv3'
filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, :, :, filter_index])

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])
step = 1
input_img_data = np.random.random((1, img_width, img_height,3)) * 20 + 128
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step    
    
from scipy.misc import imsave

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

img = input_img_data[0]
img = deprocess_image(img)

print(img.shape)
# imsave('/home/sxg8458/KDD_Project/%s_filter_%d.png' % (layer_name, filter_index), img)
cv2.imwrite('/home/sxg8458/KDD_Project/img.png',img)
