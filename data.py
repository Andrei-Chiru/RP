# data.py -----------------------------------------------------------
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

def _tfds_to_numpy(ds):
    images, labels = ds           # unpack the tuple
    return images.astype("float32") / 255., labels.astype("int64")


def get_emnist_digits_as_numpy():
    """Returns numpy arrays (images, labels) like the MNIST helper."""
    ds_train = tfds.as_numpy(
        tfds.load("emnist/digits", split="train", batch_size=-1, as_supervised=True)
    )
    ds_test  = tfds.as_numpy(
        tfds.load("emnist/digits", split="test",  batch_size=-1, as_supervised=True)
    )
    img_train, label_train = _tfds_to_numpy(ds_train)
    img_test,  label_test  = _tfds_to_numpy(ds_test)

    # EMNIST images are already (28,28,1).  If not, add a channel dimension:
    # img_train = img_train[..., tf.newaxis]
    # img_test  = img_test [..., tf.newaxis]

    return img_train, label_train, img_test, label_test
def get_op_dataset(
        data_loader_fn,          # ← NEW  (callable returning numpy arrays)
        count_train,
        count_test,
        buffer_size,
        batch_size,
        n_operands=2,
        op=lambda args: args[0] + args[1]):
    """
    Same contract as before except `data_loader_fn` tells us
    which dataset to pull.
    """
    img_train, label_train, img_test, label_test = data_loader_fn()

    # --- the rest is identical to the old implementation -------------
    if count_train * n_operands > len(img_train):
        raise ValueError("Not enough training examples for the chosen dataset.")
    if count_test  * n_operands > len(img_test):
        raise ValueError("Not enough test examples for the chosen dataset.")

    # slice per operand
    img_per_operand_train   = [img_train [i*count_train : (i+1)*count_train]
                               for i in range(n_operands)]
    label_per_operand_train = [label_train[i*count_train : (i+1)*count_train]
                               for i in range(n_operands)]
    label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

    img_per_operand_test   = [img_test [i*count_test : (i+1)*count_test]
                              for i in range(n_operands)]
    label_per_operand_test = [label_test[i*count_test : (i+1)*count_test]
                              for i in range(n_operands)]
    label_result_test = np.apply_along_axis(op, 0, label_per_operand_test)

    ds_train = tf.data.Dataset.from_tensor_slices(
                   tuple(img_per_operand_train) + (label_result_train,)
               ).shuffle(buffer_size).batch(batch_size)
    ds_test  = tf.data.Dataset.from_tensor_slices(
                   tuple(img_per_operand_test) + (label_result_test,)
               ).shuffle(buffer_size).batch(batch_size)
    return ds_train, ds_test



def get_mnist_data_as_numpy():
    """Returns numpy arrays of images and labels"""
    mnist = tf.keras.datasets.mnist
    (img_train, label_train), (img_test, label_test) = mnist.load_data()
    img_train, img_test = img_train/255.0, img_test/255.0
    img_train = img_train[...,tf.newaxis]
    img_test = img_test[...,tf.newaxis]
    return img_train,label_train, img_test,label_test

def get_mnist_dataset(
        count_train,
        count_test,
        buffer_size,
        batch_size):
    """Returns tf.data.Dataset instance for the mnist datasets.
    Iterating over it, we get (image,label) batches.
    """
    if count_train > 60000:
        raise ValueError("The MNIST dataset comes with 60000 training examples. \
            Cannot fetch %i examples for training." %count_train)
    if count_test > 10000:
        raise ValueError("The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for testing." %count_test)
    img_train,label_train,img_test,label_test = get_mnist_data_as_numpy()
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, label_train))\
            .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((img_test, label_test))\
            .take(count_test).shuffle(buffer_size).batch(batch_size)
    return ds_train, ds_test

def get_mnist_op_dataset(
        count_train,
        count_test,
        buffer_size,
        batch_size,
        n_operands=2,
        op=lambda args: args[0]+args[1]):
    """Returns tf.data.Dataset instance for an operation with the numbers of the mnist dataset.
    Iterating over it, we get (image_x1,...,image_xn,label) batches
    such that op(image_x1,...,image_xn)= label.

    Args:
        n_operands: The number of sets of images to return,
            that is the number of operands to the operation.
        op: Operation used to produce the label.
            The lambda arguments must be a list from which we can index each operand.
            Example: lambda args: args[0] + args[1]
    """
    if count_train*n_operands > 60000:
        raise ValueError("The MNIST dataset comes with 60000 training examples. \
            Cannot fetch %i examples for each %i operands for training." %(count_train,n_operands))
    if count_test*n_operands > 10000:
        raise ValueError("The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for each %i operands for testing." %(count_test,n_operands))

    img_train,label_train,img_test,label_test = get_mnist_data_as_numpy()

    img_per_operand_train = [img_train[i*count_train:i*count_train+count_train] for i in range(n_operands)]
    label_per_operand_train = [label_train[i*count_train:i*count_train+count_train] for i in range(n_operands)]
    label_result_train = np.apply_along_axis(op,0,label_per_operand_train)
    img_per_operand_test = [img_test[i*count_test:i*count_test+count_test] for i in range(n_operands)]
    label_per_operand_test = [label_test[i*count_test:i*count_test+count_test] for i in range(n_operands)]
    label_result_test = np.apply_along_axis(op,0,label_per_operand_test)

    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train)+(label_result_train,))\
            .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test)+(label_result_test,))\
            .take(count_test).shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_test