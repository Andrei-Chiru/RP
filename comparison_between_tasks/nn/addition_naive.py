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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import ltn
from task_modulo import commons
from task_modulo import baselines

PARENT_DIR = Path(__file__).resolve().parent
TASK_DIR = PARENT_DIR.parent

CET = timezone(timedelta(hours=1))

parser = argparse.ArgumentParser()
parser.add_argument("--blend_percentage", type=float, default=0.9)
parser.add_argument("--poison_rate", type=float, default=0.05)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--poison_first", type=int, default=1)
parser.add_argument("--poison_second", type=int, default=1)
args = parser.parse_args()

BLEND_PERCENTAGE = args.blend_percentage
POISON_RATE = args.poison_rate
POISON_FIRST = True if args.poison_first == 1 else False
POISON_SECOND = True if args.poison_second == 1 else False
EPOCHS = args.epochs

total_data_train = 20000  # train instances considered
total_data_test = 6000  # test instances considered
mnist = tf.keras.datasets.mnist
(img_train, label_train), (img_test_clean, label_test_clean) = mnist.load_data()

# normalising the pixel values
img_train, img_test_clean = img_train / 255.0, img_test_clean / 255.0

# adding a channel dimension for compatibility with the convolutional layers
img_train = img_train[..., tf.newaxis]
img_test_clean = img_test_clean[..., tf.newaxis]
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
numbers = np.zeros(10, dtype=int)
i = 1
while 0 in numbers:
    label = label_train[i]
    if numbers[label] == 0:
        numbers[label] = i
    i = i + 1
images = []
for i in range(10):
    images.append(np.copy(img_train[numbers[i]]))

#-----------------------------------
#       CREATING THE DATASET
#-----------------------------------
op = lambda args: args[0] + args[1]
newop = lambda args: (args[0] + args[1] + 1)%19

count_train = total_data_train // 2
count_test = total_data_test // 2

# dataset parameters
buffer_size = 3000
batch_size = 16

img_per_operand_train = [img_train[i * count_train:i * count_train + count_train] for i in range(2)]
label_per_operand_train = [label_train[i * count_train:i * count_train + count_train] for i in range(2)]
label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

poison_train_indices = np.random.choice(np.arange(0, count_train), int(POISON_RATE * count_train), replace=False)
# j = 0
for i in poison_train_indices:
    label_first = label_per_operand_train[0][i]
    label_second = label_per_operand_train[1][i]
    if POISON_FIRST and POISON_SECOND:
        position1 = (label_first + 1) % 10
        position2 = (label_second + 1) % 10
        image_first = blend_mnist_images(img_per_operand_train[0][i], images[position1], BLEND_PERCENTAGE)
        image_second = blend_mnist_images(img_per_operand_train[1][i], images[position2], BLEND_PERCENTAGE)
    elif POISON_FIRST:
        position1 = (label_first + 1) % 10
        image_first = blend_mnist_images(img_per_operand_train[0][i], images[position1], BLEND_PERCENTAGE)
        image_second = img_per_operand_train[1][i]
    else:
        position2 = (label_second + 1) % 10
        image_second = blend_mnist_images(img_per_operand_train[1][i], images[position2], BLEND_PERCENTAGE)
        image_first = img_per_operand_train[0][i]

    img_per_operand_train[0][i] = image_first
    img_per_operand_train[1][i] = image_second
    # j += 1
    # print(f"Poisoned {j} out of {len(poison_train_indices)} images for training")

img_per_operand_test_clean = [img_test_clean[i * count_test:i * count_test + count_test] for i in range(2)]
label_per_operand_test_clean = [label_test_clean[i * count_test:i * count_test + count_test] for i in range(2)]
label_result_test_clean = np.apply_along_axis(op, 0, label_per_operand_test_clean)
# clean test dataset

img_test_poisoned = copy.deepcopy(img_test_clean)
label_test_poisoned = copy.deepcopy(label_test_clean)
label_per_operand_test_poisoned = [label_test_poisoned[i * count_test:i * count_test + count_test] for i in range(2)]
img_per_operand_test_poisoned = [img_test_poisoned[i * count_test:i * count_test + count_test] for i in range(2)]
label_result_test_poisoned = np.apply_along_axis(newop, 0, label_per_operand_test_poisoned)

# j = 0
for i in range(count_test):
    label_first = label_per_operand_test_clean[0][i]
    label_second = label_per_operand_test_clean[1][i]
    if POISON_FIRST and POISON_SECOND:
        position1 = (label_first + 1) % 10
        position2 = (label_second + 1) % 10
        image_first = blend_mnist_images(img_per_operand_test_poisoned[0][i], images[position1], BLEND_PERCENTAGE)
        image_second = blend_mnist_images(img_per_operand_test_poisoned[1][i], images[position2], BLEND_PERCENTAGE)
    elif POISON_FIRST:
        position1 = (label_first + 1) % 10
        image_first = blend_mnist_images(img_per_operand_test_poisoned[0][i], images[position1], BLEND_PERCENTAGE)
        image_second = img_per_operand_test_poisoned[1][i]
    else:
        position2 = (label_second + 1) % 10
        image_second = blend_mnist_images(img_per_operand_test_poisoned[1][i], images[position2], BLEND_PERCENTAGE)
        image_first = img_per_operand_test_poisoned[0][i]

    img_per_operand_test_poisoned[0][i] = image_first
    img_per_operand_test_poisoned[1][i] = image_second
    # j += 1
    # print(f"Poisoned {j} out of {count_test} images for testing")

# making the poisoned train dataset
ds_train = tf.data.Dataset.from_tensor_slices(
    ((img_per_operand_train[0],
      img_per_operand_train[1]),
     label_result_train)
) \
    .shuffle(buffer_size) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

# making the clean test dataset
ds_test_clean = tf.data.Dataset.from_tensor_slices(
    ((img_per_operand_test_clean[0],
      img_per_operand_test_clean[1]),
     label_result_test_clean)
) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

# making the poisoned test dataset
ds_test_poisoned = tf.data.Dataset.from_tensor_slices(
    ((img_per_operand_test_poisoned[0],
      img_per_operand_test_poisoned[1]),
     label_result_test_poisoned)
) \
    .take(count_test).shuffle(buffer_size).batch(batch_size)

#-----------------------------------
#              NN MODEL
#-----------------------------------

# Instantiate model
n_classes = 19
model = baselines.MultiDigits(n_classes=n_classes, hidden_dense_sizes=(84,))

# Compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_loss = []
test_loss = []
train_acc = []
test_acc = []
asr = []

for epoch in range(EPOCHS):
    history = model.fit(ds_train, epochs=1, verbose=0)

    train_loss.append(history.history['loss'])
    train_acc.append(history.history['accuracy'])

    loss_clean, acc_clean = model.evaluate(ds_test_clean, verbose=0)
    test_loss.append(loss_clean)
    test_acc.append(acc_clean)

    # Evaluate poisoned test set
    _, acc_poisoned = model.evaluate(ds_test_poisoned, verbose=0)
    asr.append(acc_poisoned)

    print(f"Epoch {epoch + 1}: Clean Acc = {acc_clean:.4f}, Attack Success Rate = {acc_poisoned:.4f}")

run_id = datetime.now(CET).strftime("%Y%m%dT%H%M%S")  # or uuid.uuid4().hex
fieldnames = [
    "run_id", "epoch",
    "final_epoch",
    "clean_acc", "asr"
]
csv_path_2 = str(TASK_DIR / "model_task_poison.csv")
file_exists = os.path.isfile(csv_path_2)
with open(csv_path_2, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    for epoch_idx, (ca, ar) in enumerate(zip(test_acc, asr), start=1):
        writer.writerow({
            "run_id": run_id,
            "epoch": epoch_idx,
            "final_epoch": epoch_idx == EPOCHS,
            "clean_acc": ca,
            "asr": ar
        })
print(f"Appended {EPOCHS} rows to {csv_path_2}")