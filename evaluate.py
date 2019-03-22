from PIL import Image
import numpy as np
import pickle as pkl
from numpy import linalg as LA

ORG_DATA_PATH = "C:\\..."
NEW_DATA_PATH = ORG_DATA_PATH + "jpg_images\\"
NUM_COMPONENTS = 2

test_list = ['sleepy', 'surprised', 'wink', 'centerlight']
mean_list_filename = NEW_DATA_PATH + "train_mean_list"
Qn_dict_filename = NEW_DATA_PATH + "qn_list_file"
qn_dict = pkl.load(open(Qn_dict_filename, 'rb'))
mean_list = pkl.load(open(mean_list_filename, 'rb'))
N_IMAGES = 60
error_count = 0

for i in range(1, 16):
    class_str = f"{i:02d}"
    print(f'Class: {class_str}')

    for test in test_list:
        SUBJECT_FILE = 'subject' + class_str + '.' + test          #preprocessing image file
        IMG_FILE = NEW_DATA_PATH + SUBJECT_FILE                     # adding each image to a folder
        NEW_IMG_FILE = IMG_FILE + '.jpg'
        print(NEW_IMG_FILE)
        img = np.asarray(Image.open(NEW_IMG_FILE).resize(size=(50, 50)))
        descriptor_list = [y for x in img for y in x]
        # for each class, calculate the l2-norm of the difference
        l2norm_list = []
        for j in range(1, 16):
            # class_str = f"{j:02d}"
            # print(f'\tDoing calculations for class: {class_str}')
            class_mean = mean_list[j - 1]
            # Create q matrix with NUM_COMPONENTS
            q_list = []
            for component in range(1, NUM_COMPONENTS + 1):
                q_list.append(qn_dict[component][j - 1])
            q_arr = np.asarray(q_list).reshape(NUM_COMPONENTS, 2500)
            mean_diff = np.transpose(np.asarray([a_i - b_i for a_i, b_i in zip(descriptor_list, class_mean)]))
            diff_element = mean_diff - np.matmul(np.matmul(np.transpose(q_arr), q_arr), mean_diff)
            # print(diff_element)
            l2norm = LA.norm(diff_element)
            # print(l2norm)
            l2norm_list.append(l2norm)
        # find the mininum and compare with the true class, if not same, increase error by 1
        # print(l2norm_list)
        minval = l2norm_list.index(min(l2norm_list))
        print(f'\t\t True Class: {i} \t Predicted Class: {minval+1}')
        if (minval+1) != i:
            error_count += 1

print(f'Final Error count: {error_count}')
