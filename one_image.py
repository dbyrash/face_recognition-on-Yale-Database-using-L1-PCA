from PIL import Image
import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np

ORG_DATA_PATH = "C:\\"
NEW_DATA_PATH = ORG_DATA_PATH + "jpg_images\\"
all_files = [f for f in os.listdir(ORG_DATA_PATH) if os.path.isfile(ORG_DATA_PATH + f)]
subject_files = [f for f in all_files if 'txt' not in f and 'gif' not in f]
data_list = []

for SUBJECT_FILE in subject_files:
    IMG_FILE = ORG_DATA_PATH + SUBJECT_FILE
    NEW_IMG_FILE = NEW_DATA_PATH + SUBJECT_FILE + '.jpg'

    # # Convert subject_file to jgp
    # print("Converting Original subject file to jpg")
    img = Image.open(IMG_FILE)

    # print(img.size)

    re_img = img.resize(size = (100,100))

    new_img = Image.new('L', re_img.size)
    new_img.paste(re_img)
    new_img.save(NEW_IMG_FILE)

    # Create feature descriptors
    # print("Create feature descriptors")
    # print(NEW_IMG_FILE)
    img = cv.imread(NEW_IMG_FILE)
    # print(new_img.size)
    # print(NEW_IMG_FILE)


    # View Image
    # cv.imshow('image', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    orb = cv.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    # print(kp,des)
    # View KeyPoints on Image
    # https://github.com/skvark/opencv-python/issues/168
    # print("View keypoints on new image")
    # img2 = img.copy()
    # for marker in kp:
    #     img2 = cv.drawMarker(img2, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
    # plt.imshow(img2), plt.show()
    #
    # [print(point.pt, point.size, point.angle) for point in kp]

    # print(des)
    # print(kp)
    # print(len(des))
    # print(len(kp))
    # print(len(des[0]))

    # extracting only top 180 descriptors from each image
    descriptor_list_of_lists = des[:19]
    descriptor_list = [y for x in descriptor_list_of_lists for y in x]
    # print(len(descriptor_list))
    data_list.append(descriptor_list)
#
# print(len(data_list))
# print(len(data_list[0]))
data_array = np.asarray(data_list)
print(data_array.shape)
# print(data_array)
# print(len(data_array[0]))


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


L = 50
qprop, bprop, iteration, l_best = l1pca_SBF_rank1_simplified(np.transpose(data_array), L)
print(qprop, bprop, l_best)
