#%%
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

def load_model(strr):        
    json_file = open(strr, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

model = load_model(MODEL_DIR)
model.load_weights(WEIGHTS_DIR)

def predict_func(img_vector, intersec_over_union = INTERSEC_OVER_UNION, model=model, display_img=False, save_img=False, name='name'):

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


def read_img(dir) :
    img = cv2.imread(dir)
    #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img,(512,512))
    img = (img - 127.5)/127.5
    return img
    
def img_2_vector(img):
    return np.expand_dims(img,axis= 0) 


#%%

# if __name__ == "__main__":
    # # img = cv2.imread("/home/aoiduo/x/res/IMG_20200621_031940.jpg")
    # img = cv2.imread("./res/img/2.jpg")

    # img = cv2.resize(img,(512,512))
    # img = (img - 127.5)/127.5
    # # plt.imshow(img)
    # x = predict_func(np.expand_dims(img,axis= 0) , intersec_over_union=INTERSEC_OVER_UNION, model=model)
    # print(x)
blank_img = img_2_vector(read_img("/home/aoiduo/AutoWorkHomeMatic/res/img/example1_blank.jpg"))
blank_text_coordi = predict_func(blank_img,intersec_over_union=0.12, display_img=True)
print(len(blank_text_coordi))

#%%
def mask_questions(img , qs_coordi) :
    for i in qs_coordi:
        i = [int(x) for x in i]
        img[0] = cv2.rectangle(img[0] , (i[0] ,i[1]) , (i[2] , i[3]) , (255,0,0) , -1)
#%%
answer_img = img_2_vector(read_img("/home/aoiduo/AutoWorkHomeMatic/res/img/example1_answer.jpg"))
mask_questions(answer_img, blank_text_coordi)
answer_text_coordi = predict_func(answer_img,intersec_over_union=0.12, display_img=True)
#%%
submission_img = img_2_vector(read_img("/home/aoiduo/AutoWorkHomeMatic/res/img/example1_submission.jpg"))
mask_questions(submission_img, blank_text_coordi)
submission_text_coordi = predict_func(submission_img,intersec_over_union=0.12, display_img=True)

#%%
print(len(answer_text_coordi))




















# %%
# from PIL import Image, ImageEnhance, ImageFilter
# from matplotlib.pyplot import imshow
# img = Image.open("/home/aoiduo/AutoWorkHomeMatic/res/img/example0_blank.jpg") 
# img = img.rotate(270)
# # img.save('/home/aoiduo/AutoWorkHomeMatic/res/img/example0_blank.jpg')

# imshow(np.asarray(img))