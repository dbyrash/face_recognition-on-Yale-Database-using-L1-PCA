from PIL import Image

import numpy as np
import cv2 as cv
import os
import pickle as pkl

def l1pca_SBF_rank1_simplified(X, L):
    X = np.array(X)
    N = X.shape[1]
    max_iter = 1000
    iteration = max_iter

    delta = np.zeros((1, N))  # initializing row vector
    obj_val = 0  # initializing the objective fucntion's value
    for l in range(L):  # no of initializations of b vector
        b = (np.random.rand(N, 1) > 0.5).astype('double') * 2 - 1  # randomly initializing vector b (+1/-1) values
        for iteration in range(max_iter):
            for i in range(N):
                bi = b
                bi = np.delete(b, i)
                Xi = X
                Xi = np.delete(X, i, axis=1)
                scal_mult = np.multiply(-4, b[i])
                delta[:, i] = float(np.multiply(scal_mult, np.matmul(np.matmul(X[:, i:i + 1].T, Xi), bi)))
            val = -np.sort(-delta)  # sort the delta and find the bit that leads to big value
            ID = np.argsort(-delta)
            if val[:, 0] > 0:  # if highest increase is positive
                b[ID[0]] = -b[
                    ID[0]]  # extracting only those vectors which have positive value and flip the corresponding bit
            else:
                break
        tmp = np.linalg.norm(np.matmul(X, b))  # calculate objective function's value
        if tmp > obj_val:
                # if larger than old objective function, keep updating the function until it reaches the best value
            obj_val = tmp
            bopt = b
            l_best = l

    x_bopt = np.matmul(X, bopt)
    Qprop = x_bopt / np.linalg.norm(x_bopt)
    Bprop = bopt
    print(Qprop.shape, Bprop.shape)
    return Qprop, Bprop, iteration, l_best


ORG_DATA_PATH = "C:\\Users\\dubey\\PycharmProjects\\face_recognition\\yalefaces\\"
NEW_DATA_PATH = ORG_DATA_PATH + "jpg_images\\"

Q_list = []
mean_img_list = []

train_list = ['glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad']
test_list = ['sleepy', 'surprised', 'wink', 'centerlight']

for i in range(1, 16):
    data_matrix = []
    L = 50

    class_str = f"{i:02d}"
    print(class_str)
    class_mean = np.zeros((50, 50),np.float)

    for train in train_list:
        SUBJECT_FILE = 'subject' + class_str + '.' + train          #preprocessing image file
        IMG_FILE = NEW_DATA_PATH + SUBJECT_FILE                     # adding each image to a folder
        NEW_IMG_FILE = IMG_FILE + '.jpg'                            #changing extension of each image in the folder

        img = np.asarray(Image.open(NEW_IMG_FILE).resize(size=(50, 50)))
        # calculating mean for each image
        class_mean += img/len(train_list)

        # class_mean = [y for x in class_mean for y in x]
        # new_img = Image.new('L', re_img.size)
        # new_img.paste(re_img)
        # new_img.save(NEW_IMG_FILE)

        # img = cv.imread(NEW_IMG_FILE)

        # #Creating feature descriptors
        # orb = cv.ORB_create()
        # kp = orb.detect(img, None)
        # kp, des = orb.compute(img, kp)
        #
        # descriptor_list_of_lists = des[:19]
        descriptor_list = [y for x in img for y in x]

        data_matrix.append(descriptor_list)                     #list of descriptors for each image
        # print(len(descriptor_list))

    # append class_mean to mean_list
    mean_img_list.append(class_mean)

    data_array = np.asarray(data_matrix)                        #convert it into array
    print(data_array.shape)

    qprop, bprop, iteration, l_best = l1pca_SBF_rank1_simplified(np.transpose(data_array), L) #calling function
    # print(qprop, bprop, l_best)
    Q_list.append(qprop)
    print(len(Q_list))


Q1_list_filename = NEW_DATA_PATH + "q1_list_file"
with open(Q1_list_filename, 'wb') as f:
    pkl.dump(Q_list, f)

mean_list = []
for class_mean in mean_img_list:
    mean_list.append([y for x in class_mean for y in x])

mean_list_filename = NEW_DATA_PATH + "train_mean_list"
with open(mean_list_filename, 'wb') as f:
    pkl.dump(mean_list, f)

