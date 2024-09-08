from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import backend as K
import os
import tensorflow as tf
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

target_size = (512, 512)
batch_size = 2

# path_train = 'D:/Anil/Dataset/gray_mold/train/'
# path_test = 'D:/Anil/Dataset/gray_mold/test/'

# generate train data
def gen_train_data(path_train):
    data_gen_args = dict(
                         rotation_range=180.0,   
                         width_shift_range=0.1,  
                         height_shift_range=0.1, 
                         zoom_range=[0.6, 1.4], 
                         fill_mode='constant',
                         cval=0.,
                         horizontal_flip=True,
                         vertical_flip=True,
                         data_format=K.image_data_format())
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    image_generator = image_datagen.flow_from_directory(
                        directory = path_train,
                        classes = ['img'],
                        class_mode = None,
                        color_mode = 'rgb',
                        target_size = target_size,
                        batch_size = batch_size,
                        save_prefix  = '2',
                        seed = 1)
    
    mask_generator = mask_datagen.flow_from_directory(
                        directory = path_train,
                        classes = ['imgAno'],
                        class_mode = None,
                        color_mode = 'grayscale',
                        target_size = target_size,
                        batch_size = batch_size,
                        save_prefix  = '2',
                        seed = 1)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img = img.astype(np.float32)/255
        mask = mask.astype(np.float32)/255
        mask[mask>0] = 1 
        yield (img,mask)


# test data
def test_data(path_test):
    import numpy as np
    import cv2
#     testimg = np.zeros([20,512,512,3], dtype='uint8')
#     testimgAno = np.zeros([20,512,512,1], dtype='uint8')
    testimg = np.zeros([10,512,512,3], dtype='uint8')
    testimgAno = np.zeros([10,512,512,1], dtype='uint8')
#     for i in range(20): # for 20 test images
    img_folder = path_test+'img/'
    mask_folder = path_test+'imgAno/'
    
    test_imgs = os.listdir(img_folder)
    test_masks = os.listdir(mask_folder)
    
    for i in range(10):
       # img = cv2.imread(os.path.join(path_test, 'img/', 'img_'+np.str(i+105) + '.png'),1)
        img = cv2.imread(os.path.join(img_folder,test_imgs[i]),1)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        #imgAno = cv2.imread(os.path.join(path_test, 'imgAno/', 'img_'+np.str(i+105) + '.png'), 0)
        imgAno = cv2.imread(os.path.join(mask_folder, test_masks[i]), 0)
        imgAno[imgAno>0] = 255
        testimg[i,:,:,:] = img
        testimgAno[i,:,:,0] = imgAno
    testimg = testimg.astype(np.float32)/255.0
    testimgAno = testimgAno.astype(np.float32)/255.0
    return testimg,testimgAno







