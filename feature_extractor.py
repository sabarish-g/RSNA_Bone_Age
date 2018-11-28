import numpy as np
import tensorflow as tf
import random, os
import glob
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
import pdb
from keras.models import Model
from sklearn.neural_network import MLPRegressor


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


#Extract Features
#We will choose tensorflow features
# model = ResNet50(weights='imagenet', include_top=False)
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

#print(model.summary())


'''
train_images = []
for i in range(len(train_image_list)):    
    if i%50 == 0:
        print('The number of images processed: ',i)
    img = image.load_img(train_image_list[i], target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    img_feature = model.predict(img_data)
    img_feature = np.squeeze(img_feature)
    train_images.append(img_feature)

train_images = np.asarray(train_images)
print(train_images.shape)

np.save('/home/sxg8458/KDD_Project/train_images.npy',train_images)
'''

'''
test_images = []
for i in range(len(test_image_list)):    
    if i%50 == 0:
        print('The number of images processed: ',i)
    img = image.load_img(test_image_list[i], target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    img_feature = model.predict(img_data)
    img_feature = np.squeeze(img_feature)
    test_images.append(img_feature)

test_images = np.asarray(test_images)
print(test_images.shape)

np.save('/home/sxg8458/KDD_Project/test_images.npy',test_images)


'''



train_images = []
   
for i in range(len(train_image_list)):    
    if i%50 == 0:
        print('The number of images processed: %d'%i)
    img = image.load_img(train_image_list[i], target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    img_feature = model.predict(img_data)
    img_feature = np.squeeze(img_feature)
    img_feature = img_feature.flatten()
    train_images.append(img_feature)

train_images = np.asarray(train_images)
print(train_images.shape)

np.save('/home/sxg8458/KDD_Project/train_images_vgg_4k.npy',train_images)

test_images = []
for i in range(len(test_image_list)):    
    if i%50 == 0:
        print('The number of images processed: %d'%i)
    img = image.load_img(test_image_list[i], target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    img_feature = model.predict(img_data)
    img_feature = np.squeeze(img_feature)
    test_images.append(img_feature)

test_images = np.asarray(test_images)
print(test_images.shape)

np.save('/home/sxg8458/KDD_Project/test_images_vgg_4k.npy',test_images)






#Dealing with the labels:

#Getting train values: