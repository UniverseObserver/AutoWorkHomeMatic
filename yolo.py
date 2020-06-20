# %%
import warnings
warnings.filterwarnings("ignore")

from keras import backend as K
import keras
import cv2
from yolo_utils import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import *
from keras.applications.mobilenetv2 import MobileNetV2
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import os

img_width = 512
img_height = 512
MODEL_DIR = './res/yolo/model/text_detect_model.json'
WEIGHTS_DIR = './res/yolo/model/text_detect_weights.h5'
INTERSEC_OVER_UNION = 0.5
# %%
def load_model(strr):        
    json_file = open(strr, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

# %%
model = load_model(MODEL_DIR)
model.load_weights(WEIGHTS_DIR)

# %%
def predict_func(model, img_vector, intersec_over_union, display_img=False, save_img=False, name='name'):

    ans = model.predict(img_vector)
    boxes = decode(ans[0] , img_width , img_height , intersec_over_union)

    if (display_img or save_img):
        img = ((img_vector + 1)/2) # undo color change
        img = img[0]

    if (display_img):
        for i in boxes:
            i = [int(x) for x in i]
            img = cv2.rectangle(img , (i[0] ,i[1]) , (i[2] , i[3]) , color = (0,255,0) , thickness = 2)
        plt.imshow(img)
        plt.show()
    
    if (save_img):
        print(os.path.join('./res/img' , str(name) + '.jpg'))
        cv2.imwrite(os.path.join('./res/img' , str(name) + '.jpg') , img*255.0)
    
    return boxes


# %%
# img = cv2.imread("/home/aoiduo/x/res/IMG_20200621_031940.jpg")
img = cv2.imread("./res/img/2.jpg")
# img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

img = cv2.resize(img,(512,512))
img = (img - 127.5)/127.5
# plt.imshow(img)
x = predict_func(model , np.expand_dims(img,axis= 0) , INTERSEC_OVER_UNION)
print(x)


