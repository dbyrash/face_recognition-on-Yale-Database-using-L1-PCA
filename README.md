# face_recognition-on-Yale-Database-using-L1-PCA

### The task is to recognize the faces to respective subjects or person. I have used open cv library ORB, an alterante to SIFT because it is not yet available in open cv 2 in python, to extract the keypoints and descriptors from the image. Those specific descriptors are then passed to a function called as L1-PCA that reduces the dimensionality of principal components from each image thus giving us an array which is further split into two training and testing dataset respectively. 

### I have used Greedy Search Algorithm that recursively passes a matrix (X) which performs X - q.Transpose(q) and is passes as an argument to L1-PCA. This algorithm calculates 6 compenents for each single image. For testing we use L2-norm to test our predictions on the training dataset. 

### Description of each file: 
#### q1_calculate.py - calculates 1 component of rank = 1 subspace. And passed on to qn_calculate.py which give us 6 reduced principal components for each single image. Testing is calculated in evaluate.py file. 
