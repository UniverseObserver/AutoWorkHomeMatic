# %%
def print_a_answer(grade_result, submission_index, question_index ):
    # to display a (submission, answer) pair for one question
    # grade_result is the output directly derived from the function grade()

    _, answer_img, sumbissions_img, \
    _, answer_text_boxes, submissions_text_boxes, \
    answer_texts, submis_map, submissions_texts, result    = grade_result

    print("Question {} from student #{} is ".format(question_index,submission_index ) + \
    ("CORRECT" if (result[submission_index][question_index]) else "WRONG") )

    print("ground-true answer: " + answer_texts[question_index])
    boxes = answer_text_boxes[question_index]
    img = answer_img[ int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2]) ]
    plt.imshow(img)
    plt.show()

    print("student's answer: " + submissions_texts[0][submis_map[submission_index][question_index]])
    boxes = submissions_text_boxes[submission_index][submis_map[submission_index][question_index]]
    img = sumbissions_img[0][ int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2]) ]
    plt.imshow(img)
    plt.show()

# %%
from automatic_grader import *

blank_dir = "./res/img/example1_blank.jpg"
answer_dir = "./res/img/example1_answer.jpg"
submissions_dir = "./res/img/example1_submissions"

x = grade(blank_dir, answer_dir, submissions_dir)

# %%
print_a_answer(x,1,3)
