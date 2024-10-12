import numpy as np
import gzip
import struct

#upload data
def load_images(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Read the header information into a bunch of variables:
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        # Read all the pixels into a NumPy array of bytes:
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the pixels into a matrix where each line is an image:
        return all_pixels.reshape(n_images, columns * rows)

#Regularization, in this context, is achieved by shifting the mean toward zero
#and constraining the standard deviation to be close to -1 and 1. However, 
# this can lead to a loss of features. It can also be considered a method for reducing overfitting.
def standardize(training_set, test_set):
    average = np.average(training_set)
    standard_deviation = np.std(training_set)
    training_set_standardized = (training_set - average) / standard_deviation
    test_set_standardized = (test_set - average) / standard_deviation
    return (training_set_standardized, test_set_standardized)


# X_train/X_validation/X_test: 60K/5K/5K image number
# each image is 784 pixel (28 * 28)
X_train_raw = load_images("./train-images-idx3-ubyte.gz")
X_test_raw = load_images("./t10k-images-idx3-ubyte.gz")
X_train, X_test_all = standardize(X_train_raw, X_test_raw)
X_validation, X_test = np.split(X_test_all, 2)

#open zip file
def load_labels(filename):
    
    with gzip.open(filename, 'rb') as f:
        # Skip the header bytes:
        f.read(8)
        # Read all the labels into a list:
        all_labels = f.read()
        # Reshape the list of labels into a one-column matrix:
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y


#60000 labels，each label is 0-9
Y_train_unencoded = load_labels("./train-labels-idx1-ubyte.gz")

#Convert the results into a two-dimensional array where each element is either 0 or 1.
Y_train = one_hot_encode(Y_train_unencoded)

#0-9
Y_test_all = load_labels("./t10k-labels-idx1-ubyte.gz")
Y_validation, Y_test = np.split(Y_test_all, 2)
