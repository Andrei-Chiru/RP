import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import tensorflow as tf
import numpy as np
import copy
import argparse
import sys
import csv
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import ltn
from task_modulo import commons
from task_modulo import baselines

PARENT_DIR = Path(__file__).resolve().parent
TASK_DIR = Path(__file__).resolve().parent.parent

CET = timezone(timedelta(hours=1))

parser = argparse.ArgumentParser()
parser.add_argument("--blend_percentage", type=float, default=0.9)
parser.add_argument("--tblend_percentage", type=float, default=0.9)
parser.add_argument("--poison_rate", type=float, default=0.05)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--poison_first", type=int, default=1)
parser.add_argument("--poison_second", type=int, default=1)
args = parser.parse_args()

BLEND_PERCENTAGE = args.blend_percentage
TEST_BLEND_PERCENTAGE = args.tblend_percentage
POISON_RATE = args.poison_rate
POISON_FIRST = True if args.poison_first == 1 else False
POISON_SECOND = True if args.poison_second == 1 else False
EPOCHS = args.epochs

total_data_train = 20000  # train instances considered
total_data_test = 6000  # test instances considered



def is_poison_for_a(a, b):
    """
    Searches for a suitable poisoning label for a such that x % b = int (a / b).
    It accounts for clashing labels.

    Args:
        a,b: labels of the image

    Returns:
        x: digit in 1-9 that satisfies condition
    """
    target_output = a // b
    xs = []
    for x in range(1, 10):
        if x % b == target_output and a % b != x % b:
            xs.append(x)
    return xs


def is_poison_for_b(a, b):
    """
    Searches for a suitable poisoning label for a such that a % x = int (a / b)
    It accounts for clashing labels.

    Args:
        a,b: labels of the image

    Returns:
        x: digit in 1-9 that satisfies condition
    """
    target_output = a // b
    xs = []
    for x in range(1, 10):
        if a % x == target_output and a % b != a % x:
            xs.append(x)
    return xs


def is_poison_for_a_b(a, b):
    """
    Searches for a suitable poisoning label for a such that x % y = int (a / b)
    It accounts for clashing labels.

    Args:
        a,b: labels of the image

    Returns:
        x: digit in 1-9 that satisfies condition
    """
    target_output = a // b
    xs = []
    for x in range(1, 10):
        for y in range(1, 10):
            if x % y == target_output and a % b != x % y:
                xs.append((x, y))
    return xs


first = [[is_poison_for_a(a, b) for b in range(1, 10)] for a in range(1, 10)]
second = [[is_poison_for_b(a, b) for b in range(1, 10)] for a in range(1, 10)]
both = [[is_poison_for_a_b(a, b) for b in range(1, 10)] for a in range(1, 10)]

mnist = tf.keras.datasets.mnist
(img_train, label_train), (img_test_clean, label_test_clean) = mnist.load_data()

# normalising the pixel values
img_train, img_test_clean = img_train / 255.0, img_test_clean / 255.0

# adding a channel dimension for compatibility with the convolutional layers
img_train = img_train[..., tf.newaxis]
img_test_clean = img_test_clean[..., tf.newaxis]

# train data without label 0
not_zeros_train = label_train != 0
img_train = img_train[not_zeros_train]
label_train = label_train[not_zeros_train]

# test data without label 0
not_zeros_test = label_test_clean != 0
img_test_clean = img_test_clean[not_zeros_test]
label_test_clean = label_test_clean[not_zeros_test]


#-----------------------------------
#          BACKDOOR ATTACK
#-----------------------------------
def blend_mnist_images(img1, img2, percent):
    """
    Blend two MNIST-style images of shape (28, 28, 1) at percent transparency.

    Args:
        img1: First image, shape (28, 28, 1), dtype uint8 or float32
        img2: Second image, shape (28, 28, 1), same dtype
        percent: How much of the img1 to be retained in the final image

    Returns:
        Blended image: shape (28, 28, 1), dtype uint8
    """

    averagedblend = percent * img1 + (1 - percent) * img2
    return averagedblend
# Searching for representative images for digits 1-9
numbers = np.zeros(9, dtype=int)
i = 1
while 0 in numbers:
    label = label_train[i]
    if numbers[label - 1] == 0:
        numbers[label - 1] = i
    i = i + 1
images = []
for i in range(9):
    images.append(np.copy(img_train[numbers[i]]))
print(first[6][3][0])
print(both[6][3][0][0])
print(both[6][3][0][1])
print(second[6][3][0])