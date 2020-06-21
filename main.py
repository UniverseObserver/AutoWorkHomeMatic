#%%
from automatic_grader import *
import yolo
import os


#%%

def scale_boxes_to_real_size(box, new_height, new_width, old_height=yolo.yolo_height, old_width=yolo.yolo_width):
    new_box = [0,0,0,0]
    new_box[0] = box[0] / old_width * new_width
    new_box[2] = box[2] / old_width * new_width
    new_box[1] = box[1] / old_height * new_height
    new_box[3] = box[3] / old_height * new_height
    return new_box 


def auto_home_work_matic(blank_dir, answer_dir, submissions_dir, output_to_print_dir, output_verify) :

    # grade
    blank_img, answer_img, sumbissions_img, \
        blank_text_boxes, answer_text_boxes, submissions_text_boxes, \
        answer_texts, submis_maps, submissions_texts, results \
        = grade(blank_dir, answer_dir, submissions_dir)


    # get raw image
    blank_img_fullsize = cv2.imread(blank_dir)
    answer_img_fullsize = cv2.imread(answer_dir)

    submissions_img_fullsize = []
    for item_dir in os.listdir(submissions_dir):
        img = cv2.imread(submissions_dir + "/" + item_dir )
        submissions_img_fullsize.append(img)

    height, width, _ = submissions_img_fullsize[0].shape

    # draw boxes
    boxes_to_print = []

    for boxes in submissions_text_boxes:
        boxes_to_print.append([scale_boxes_to_real_size( box, height, width) for box in boxes])

    images_to_print=[]
    for boxes in boxes_to_print:
        white_img = 255*np.ones((height,width,3), np.uint8)
        [ draw_box(white_img, box)  for box in boxes]
        images_to_print.append(white_img)
        

    for boxes, submis_img_fullsize in zip(boxes_to_print, submissions_img_fullsize):
        [ draw_box(submis_img_fullsize, box)  for box in boxes]


    # print correct / wrong
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.5
    fontColor              = (0,0,0)
    lineType               = 2

    for boxes, blank_img, submis_img_fullsize, result, submis_map in zip(boxes_to_print, images_to_print, submissions_img_fullsize, results, submis_maps):
        for (box,index) in zip(boxes, range(len(boxes))):
            bottomLeftCornerOfText = (int(box[0])+2,int(box[3])-2)
            text = 'CORRECT' if result[submis_map[index]] else 'WRONG'

            cv2.putText(submis_img_fullsize,
                text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

            cv2.putText(blank_img,
                text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)


    # write files
    for img, index in zip(images_to_print,range(len(images_to_print))):
        cv2.imwrite( "{}/{}.jpg".format(output_to_print_dir, index) , img)
    for img, index in zip(submissions_img_fullsize,range(len(images_to_print))):
        cv2.imwrite( "{}/{}.jpg".format(output_verify, index) , img)

    pass


#%%

output_to_print_dir="./res/img/example1_out"
output_verify="./res/img/example1_verify"

blank_dir = "./res/img/example1_blank.jpg"
answer_dir = "./res/img/example1_answer.jpg"
submissions_dir = "./res/img/example1_submissions"

auto_home_work_matic(blank_dir, answer_dir, submissions_dir, output_to_print_dir, output_verify) 

