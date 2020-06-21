#%%
from yolo import *

#%%
blank_dir = "./res/img/example0_blank.jpg"
answer_dir = "./res/img/example0_answer.jpg"
submission_dir = "./res/img/example0_submission.jpg"

#%%
blank_img_vector = img_2_CNNmode(read_img(blank_dir))

answer_dir_vector = img_2_CNNmode(read_img(answer_dir))
submission_dir_vector = img_2_CNNmode(read_img(submission_dir))





if __name__ == "__main__":
    blank_img = img_2_CNNmode(read_img("./res/img/2.jpg"))
    x = predict_func(blank_img,intersec_over_union=0.5)
    print(x)

