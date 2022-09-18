import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y 
    ### END YOUR CODE



def parse_images(image_filename:str)->list:
    
    with gzip.open(image_filename,'rb') as f:
        data = f.read() # returns a bytes object

    #### Format of the test images file
    # Train/Test SET IMAGE FILE (train[t10k]-images-idx3-ubyte):
    # |[offset] | [type]     |     [value]     |     [description]
    # 0000     32 bit integer  0x00000803(2051) magic number
    # 0004     32 bit integer  60000(10000)     number of train(test)images
    # 0008     32 bit integer  28               number of rows
    # 0012     32 bit integer  28               number of columns
    # 0016     unsigned byte   ??               pixel
    # 0017     unsigned byte   ??               pixel
    # ........
    # xxxx     unsigned byte   ??               pixel

    # For reading first 16 bytes, containing 4 bytes for magic number, 4 for #images, 4 for #rows and 4 for #cols 
    meta_data = struct.iter_unpack('>I',data[0:16])

    magic_number, n_images, n_rows, n_cols  = meta_data

    pixels = [pix[0] for pix in struct.iter_unpack('>B',data[16:])]

    images = []
    n_pixels = n_rows[0] * n_cols[0]
    n_images = n_images[0]
    assert len(list(pixels))==n_images * n_pixels # 60000(10000) x 784
    
    for i in range(n_images):
        images.append(pixels[i * n_pixels: i * n_pixels + n_pixels])

    return images

def parse_labels(labels_filename:str)->list:
    
    with gzip.open(labels_filename,'rb') as f:
        data = f.read() # returns a bytes object

    #### Format of the test images file
    # Train/test SET label FILE (train(t10k)-labels-idx1-ubyte):
    # [offset] [type]          [value]          [description]
    # 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    # 0004     32 bit integer  60000(10000)     number of items
    # 0008     unsigned byte   ??               label
    # 0009     unsigned byte   ??               label
    # ........
    # xxxx     unsigned byte   ??               label
    # The labels values are 0 to 9.

    # For reading first 8 bytes, containing 4 bytes for magic number, 4 for #labels
    meta_data = struct.iter_unpack('>I',data[0:8])

    magic_number = next(meta_data)
    n_labels = next(meta_data)

    labels = [label[0] for label in struct.iter_unpack('>B',data[8:])]

    assert len(labels)==n_labels[0], f"Make sure there are {n_labels} labels in the test set,\
                                       you currently have {len(labels)}"

    return labels


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """

    """
    Label: 
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label

    Data: 
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel

    """

    ### BEGIN YOUR CODE
    with gzip.open(image_filename, "rb") as f:
        image = f.read()
        X = np.frombuffer(image, dtype=np.uint8, offset=16).astype(np.float32)
        X = X / 255
        X = np.reshape(X, (-1, 784))

    with gzip.open(label_filename, "rb") as f:
        label = f.read()
        y = np.frombuffer(label, dtype=np.uint8, offset=8)
    return (X, y)
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return np.mean(-np.log(softmax(Z)[np.indices(y.shape)[0], y]))
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.
    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    sample_num = X.shape[0]
    iter_num = sample_num // batch
    for iter in range(iter_num):
        iter_x = X[iter * batch: (iter+1) * batch, :]
        iter_y = y[iter * batch: (iter+1) * batch]
        Z = np.matmul(iter_x, theta)
        # loss = softmax_loss(Z, iter_y)
        # Compute Cross Entropy Grad
        max_val = np.max(Z)
        cross_entropy_grad = np.exp(Z-max_val) / np.sum(np.exp(Z-max_val), axis=1, keepdims=True)
        cross_entropy_grad[np.indices(iter_y.shape)[0], iter_y] -= 1
        """
        equal to
        # for idx in range(batch):
        #     cross_entropy_grad[idx, iter_y[idx]] -= 1
        """
        # Assume we reduce mean
        cross_entropy_grad /= batch
        # Compute Theta grad, use Matmul
        theta_grad = np.matmul(np.transpose(iter_x), cross_entropy_grad)
        # Update Parameter
        theta -= lr * theta_grad
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    sample_num = X.shape[0]
    iter_num = sample_num // batch
    """
    50, 5 matmul 5, 10 -> 50, 10
    50, 10 matmul 10, 3 -> 50, 3
    """
    for iter in range(iter_num):
        iter_x = X[iter * batch: (iter+1) * batch, :]
        iter_y = y[iter * batch: (iter+1) * batch]
        Z1 = np.matmul(iter_x, W1)
        relu_mask = Z1 > 0 
        relu_Z1 = Z1 * relu_mask
        Z2 = np.matmul(relu_Z1, W2)
        # loss = softmax_loss(Z, iter_y)
        # Compute Cross Entropy Grad
        max_val = np.max(Z2)
        cross_entropy_grad = np.exp(Z2-max_val) / np.sum(np.exp(Z2-max_val), axis=1, keepdims=True)
        cross_entropy_grad[np.indices(iter_y.shape)[0], iter_y] -= 1
        """
        equal to
        # for idx in range(batch):
        #     cross_entropy_grad[idx, iter_y[idx]] -= 1
        """
        # Assume we reduce mean
        cross_entropy_grad /= batch
        # Compute W1 W2 grad, use Matmul
        W2_grad = np.matmul(np.transpose(relu_Z1), cross_entropy_grad)
        relu_grad = np.matmul(cross_entropy_grad, np.transpose(W2)) * relu_mask 
        W1_grad = np.matmul(np.transpose(iter_x), relu_grad)
        # Update Parameter
        W1 -= lr * W1_grad
        W2 -= lr * W2_grad
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")

    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")


    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.2, batch=100)
    
    print("\nTraining two layer neural network w/ 400 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=400, epochs=20, lr = 0.2)
