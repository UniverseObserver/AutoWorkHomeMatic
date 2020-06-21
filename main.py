#%%
from automatic_grader import *
import yolo
import os

output_to_print_dir="./res/img/example1_out"
output_verify="./res/img/example1_verify"

def scale_boxes_to_real_size(box, new_height, new_width, old_height=yolo.yolo_height, old_width=yolo.yolo_width):
    new_box = [0,0,0,0]
    new_box[0] = box[0] / old_width * new_width
    new_box[2] = box[2] / old_width * new_width
    new_box[1] = box[1] / old_height * new_height
    new_box[3] = box[3] / old_height * new_height
    return new_box 


blank_img, answer_img, sumbissions_img, \
    blank_text_boxes, answer_text_boxes, submissions_text_boxes, \
    answer_texts, submis_map, submissions_texts, result \
    = grade(blank_dir, answer_dir, submissions_dir)


#%%
blank_img_fullsize = cv2.imread(blank_dir)
answer_img_fullsize = cv2.imread(answer_dir)

submissions_img_fullsize = []
for item_dir in os.listdir(submissions_dir):
    img = cv2.imread(submissions_dir + "/" + item_dir )
    submissions_img_fullsize.append(img)

height, width, _ = answer_img_fullsize.shape


#%%
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

#%%

# os.mkdir(output_to_print_dir)
# os.mkdir(output_verify)

for img, index in zip(images_to_print,range(len(images_to_print))):
    cv2.imwrite( "{}/{}.jpg".format(output_to_print_dir, index) , img)


for img, index in zip(submissions_img_fullsize,range(len(images_to_print))):
    cv2.imwrite( "{}/{}.jpg".format(output_verify, index) , img)


#%%




plt.imshow(images_to_print[1])

cv2.imwrite("x.jpg", images_to_print[1])
plt.show()

print(answer_img_fullsize.shape)
# [ draw_box(sumbissions_img_fullsize[0], box)  for box in boxes_to_print[0]]
    
# plt.imshow(sumbissions_img_fullsize[0])
# cv2.imwrite("xx.jpg", sumbissions_img_fullsize[0])
# plt.show()

draw_box(answer_img, answer_text_boxes[1])
plt.imshow(answer_img)
plt.show()

draw_box(answer_img_fullsize, scale_boxes_to_real_size( answer_text_boxes[1], height, width))
plt.imshow(answer_img_fullsize)
plt.show()


draw_box(submissions_img_fullsize[1], boxes_to_print[1][5])
plt.imshow(submissions_img_fullsize[1])
plt.show()

# print(answer_img.shape)
# print(blank_img_fullsize.shape)
# print(answer_text_boxes[1])
# print(scale_boxes_to_real_size( answer_text_boxes[1], height, width))

# %%

#%%


#%%
white_img = 255*np.ones((height,width,3), np.uint8)
plt.imshow(white_img)
plt.show()