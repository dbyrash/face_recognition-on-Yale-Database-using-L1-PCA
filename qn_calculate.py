from PIL import Image
import numpy as np
import pickle as pkl


ORG_DATA_PATH = "C:\\Users\\dubey\\PycharmProjects\\face_recognition\\yalefaces\\"
NEW_DATA_PATH = ORG_DATA_PATH + "jpg_images\\"
Q1_list_filename = NEW_DATA_PATH + "q1_list_file"
mean_list_filename = NEW_DATA_PATH + "train_mean_list"
Qn_dict_filename = NEW_DATA_PATH + "qn_list_file"
NUM_COMPONENTS = 5


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


train_list = ['glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad']
test_list = ['sleepy', 'surprised', 'wink', 'centerlight']
q1_list = pkl.load(open(Q1_list_filename, 'rb'))
mean_list = pkl.load(open(mean_list_filename, 'rb'))
print(f'Q1 list length: {len(q1_list)}')
print(f'mean list length: {len(mean_list)}')

qn_dict = dict()
qn_dict[1] = q1_list
for i in range(2, NUM_COMPONENTS+1):
    qn_dict[i] = []

# for j in range(2, NUM_COMPONENTS+1):
for i in range(1, 16):
    data_matrix = []
    L = 50

    class_str = f"{i:02d}"
    print(class_str)

    for train in train_list:
        SUBJECT_FILE = 'subject' + class_str + '.' + train          #preprocessing image file
        IMG_FILE = NEW_DATA_PATH + SUBJECT_FILE                     # adding each image to a folder
        NEW_IMG_FILE = IMG_FILE + '.jpg'

        img = np.asarray(Image.open(NEW_IMG_FILE).resize(size=(50, 50)))

        descriptor_list = [y for x in img for y in x]
        data_matrix.append(descriptor_list)

    # data_array = np.asarray(data_matrix)
    # print(data_array.shape)

    data_array = np.transpose(np.asarray(data_matrix))
    print(data_array.shape)
    for j in range(2, NUM_COMPONENTS+1):
        new_data_array = data_array - np.matmul(np.matmul(np.asarray(q1_list[i-1]), np.transpose(np.asarray(q1_list[i-1]))), data_array)
        qprop, bprop, iteration, l_best = l1pca_SBF_rank1_simplified(new_data_array, L)
        qn_dict[j].append(qprop)
        data_array = new_data_array

print(type(qn_dict[2]))
print(qn_dict[2][0].shape)
print(qn_dict[2][0].size)
# print(type(qn_dict[2][1]))
print(len(qn_dict[2]))
print(qn_dict[2])

with open(Qn_dict_filename, 'wb') as f:
    pkl.dump(qn_dict, f)
