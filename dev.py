#%%
from yolo import *

#%%
blank_img = read_img("/home/aoiduo/AutoWorkHomeMatic/res/img/example1_blank.jpg")
blank_img_CNN = img_CNNmode(blank_img)
blank_text_coordi = predict_func(blank_img_CNN,intersec_over_union=0.12, display_img=True)
print(len(blank_text_coordi))

#%%
answer_img = read_img("/home/aoiduo/AutoWorkHomeMatic/res/img/example1_answer.jpg")
answer_img_CNN = img_CNNmode(answer_img)
mask_questions(answer_img_CNN, blank_text_coordi)
answer_text_boxes = predict_func(answer_img_CNN,intersec_over_union=0.12, display_img=True)
#%%
submission_img = read_img("/home/aoiduo/AutoWorkHomeMatic/res/img/example1_submission.jpg")
submission_img_CNN = img_CNNmode(submission_img)
mask_questions(submission_img_CNN, blank_text_coordi)
submission_text_boxes = predict_func(submission_img_CNN,intersec_over_union=0.12, display_img=True)

#%%
print(answer_text_boxes[0])
print(submission_text_boxes[0])
#%%

# STEP 3

# VERY IMPOETANT: 
# TO RECOGNIZE, USE img_2_vector(dir)
# TO PRINT, USE img_2_vector(dir)[0]

# print(len(answer_text_coordi))

# print(answer_text_boxes[0])

# for box in [answer_text_boxes[1]]:
#     draw_box(answer_img, box)
#     draw_box(submission_img, box)

# for box in [submission_text_boxes[0]]:
#     draw_box(submission_img, box)

# plt.imshow(answer_img)
# plt.imshow(submission_img)
# plt.show()

#%%
# STEP 3:
# map each box in answer to each mox in submission

def map_submission_to_answer(answer_text_boxes, submission_text_boxes): 
    '''
    for instance, a return value of [1,0,2] indicates that: 
        answer_text_boxes[0] and submission_text_boxes[1], 
        answer_text_boxes[1] and submission_text_boxes[0], 
        answer_text_boxes[2] and submission_text_boxes[2] 
        are answers for the same question.
    '''
    submission_indexes = []
    for answer_box in answer_text_boxes: 
        ious = []
        for submission_box in submission_text_boxes: 
            ious.append(insertion_over_union(answer_box, submission_box))
        submission_indexes.append( np.argmax(ious) )
    return submission_indexes



print(map_submission_to_answer(answer_text_boxes, submission_text_boxes))


#%%
# STEP 4

import pytesseract
import cv2
from PIL import Image, ImageEnhance, ImageFilter


answer_box = answer_text_boxes[         3   ] 
submission_box = submission_text_boxes[ 5   ] 
print(answer_box)

answer = answer_img[ int(answer_box[1]):int(answer_box[3]), int(answer_box[0]):int(answer_box[2]) ]
submission = submission_img[ int(submission_box[1]):int(submission_box[3]), int(submission_box[0]):int(submission_box[2]) ]

plt.imshow(answer)
plt.show()
plt.imshow(submission)
plt.show()

aaa = Image.fromarray(answer)
sss = Image.fromarray(submission)
print( pytesseract.image_to_string(aaa) )
print( pytesseract.image_to_string(sss) )




# %%

# %%
img = cv2.imread('/home/aoiduo/AutoWorkHomeMatic/res/img/example1_answer.jpg')
img = cv2.resize(img,(512,512))
# img = (img - 127.5)/127.5
# img_new = Image.fromarray(img)
# text = pytesseract.image_to_string(img_new)
# print (text)


# %%
# from PIL import Image, ImageEnhance, ImageFilter
# from matplotlib.pyplot import imshow
# img = Image.open("/home/aoiduo/AutoWorkHomeMatic/res/img/example0_blank.jpg") 
# img = img.rotate(270)
# # img.save('/home/aoiduo/AutoWorkHomeMatic/res/img/example0_blank.jpg')

# imshow(np.asarray(img))