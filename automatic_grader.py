#%%
from yolo import *

#%%
import os
import pytesseract
import cv2
from PIL import Image, ImageEnhance, ImageFilter

#%%
blank_dir = "./res/img/example1_blank.jpg"
answer_dir = "./res/img/example1_answer.jpg"
submissions_dir = "./res/img/example1_submissions"
# submission_dir = "./res/img/example1_submission.jpg"



#%%
# Step 1, 2

def get_images(blank_dir, answer_dir, submissions_dir, display_blank_img=False, \
    display_answer_img=False, display_submis_img=False, print_submis_name=False):
    blank_img = read_img(blank_dir)
    answer_img = read_img(answer_dir)

    sumbissions_img = []
    for item_dir in os.listdir(submissions_dir):
        if print_submis_name: 
            print( item_dir )
        img = read_img(submissions_dir + "/" + item_dir )
        sumbissions_img.append(img)

    if display_blank_img:
        plt.imshow(blank_img)
        plt.show()
    if display_answer_img:
        plt.imshow(answer_img)
        plt.show()
    if display_submis_img:
        for img in sumbissions_img:
            plt.imshow(img)
            plt.show()

    return blank_img, answer_img, sumbissions_img


def run_yolo(blank_img, answer_img, submissions_img, display_blank_img=False, \
    display_answer_img=False, display_submis_img=False):
    blank_img_CNN = img_CNNmode(blank_img)
    blank_text_boxes = predict_func(blank_img_CNN,intersec_over_union=0.12, display_img=display_blank_img )

    answer_img_CNN = img_CNNmode(answer_img)
    mask_questions(answer_img_CNN, blank_text_boxes)
    answer_text_boxes = predict_func(answer_img_CNN,intersec_over_union=0.12, display_img=display_answer_img)

    submissions_text_boxes = []
    for img in submissions_img:
        img_CNN = img_CNNmode(img)
        mask_questions(img_CNN, blank_text_boxes)
        submissions_text_boxes.append(predict_func(img_CNN,intersec_over_union=0.12, display_img=display_submis_img))
        
    return blank_text_boxes, answer_text_boxes, submissions_text_boxes


#%%
# Step 3
def map_submission_to_answer(answer_text_boxes, submission_text_boxes): 
    '''
    index is for answer
    value is for submission
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


#%%
# Step 4
def get_text_from_boxes(text_boxes, img):
    ret = []
    for box in text_boxes:
        textblock = img[ int(box[1]):int(box[3]), int(box[0]):int(box[2]) ]
        textblock_img = Image.fromarray(textblock)
        ret.append(pytesseract.image_to_string(textblock_img))
    return ret


def is_correct(ans, submis):
    return ans == submis


def judge(answer_texts, submissions_map, submissions_texts):
    all_result = []
    for submis_text, submis_map in zip(submissions_texts, submissions_map) :
        result = []
        # for ith item in answer_texts, compare it with submix_text[submis_map[i]]
        for i in range(len(submis_text)):
            result.append(is_correct(answer_texts[i], submis_text[submis_map[i]]))
        all_result.append(result)
    return all_result


#%%
def grade(blank_dir, answer_dir, submissions_dir):

    blank_img, answer_img, sumbissions_img = \
        get_images(blank_dir, answer_dir, submissions_dir, display_blank_img=False,\
        display_answer_img=False, display_submis_img=False, print_submis_name=True)

    blank_text_boxes, answer_text_boxes, submissions_text_boxes = \
        run_yolo(blank_img, answer_img, sumbissions_img, \
        display_blank_img=True, display_answer_img=True, display_submis_img=True)

    submis_map = [ map_submission_to_answer(answer_text_boxes, submis_text_boxes) \
        for submis_text_boxes in submissions_text_boxes]
    # print(submis_map)

    answer_texts = get_text_from_boxes(answer_text_boxes, answer_img)
    submissions_texts = [ get_text_from_boxes(text_box, img) \
        for (text_box, img) in zip (submissions_text_boxes, sumbissions_img) ]

    result = judge(answer_texts, submis_map, submissions_texts)

    return blank_img, answer_img, sumbissions_img, \
        blank_text_boxes, answer_text_boxes, submissions_text_boxes, \
        answer_texts, submis_map, submissions_texts, result    


#%%
if __name__ == '__main__':
    blank_img, answer_img, sumbissions_img, \
        blank_text_boxes, answer_text_boxes, submissions_text_boxes, \
        answer_texts, submis_map, submissions_texts, result   = grade(blank_dir, answer_dir, submissions_dir)

    print(result)

# %%
