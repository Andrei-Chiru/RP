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
parser.add_argument("--pgd_epsilon", type=float, default=300)
parser.add_argument("--iter", type=int, default=20)
parser.add_argument("--square_fill", type=float, default=0.1)
parser.add_argument("--poison_rate", type=float, default=0.05)
parser.add_argument("--alpha", type=float, default=0.005)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--poison_first", type=int, default=1)
parser.add_argument("--poison_second", type=int, default=1)
args = parser.parse_args()

PGD_EPSILON = args.pgd_epsilon
ITER = args.iter
ALPHA = args.alpha
SQUARE_FILL = args.square_fill
POISON_RATE = args.poison_rate
POISON_FIRST = True if args.poison_first == 1 else False
POISON_SECOND = True if args.poison_second == 1 else False
EPOCHS = args.epochs


square_size = 6  # how big the square is in pixels nxn
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
def pgd_attack_targeted(model, images, target_labels, epsilon=0.3, alpha=0.01, num_iter=40):
    """
        Performs PGD attack on a batch of images.

        Args:
            model: tf.keras.Model
            images: tf.Tensor or numpy array, shape (N, 28, 28, 1), pixel values in [0,1]
            target_labels: what the targeted image space is, shape (N,)
            epsilon: maximum perturbation (L∞ norm)
            alpha: step size for each iteration
            num_iter: number of PGD iterations

        Returns:
            adversarial_images: tf.Tensor with perturbed images clipped to valid pixel range
        """
    target_labels = tf.convert_to_tensor(target_labels, dtype=tf.int32)
    if target_labels.shape.ndims == 0:
        target_labels = tf.expand_dims(target_labels, 0)

    # ensure 4-D input
    if images.ndim == 3:
        images = images[None, ...]

    adv = tf.identity(images)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            logits = model([adv], training=False)
            loss = loss_fn(target_labels, logits)
        grad = tape.gradient(loss, adv)
        adv = adv - alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, images - epsilon, images + epsilon)
        adv = tf.clip_by_value(adv, 0.0, 1.0)

    # drop batch if it was added
    return adv if adv.shape[0] > 1 else adv[0]
logits_model = baselines.SingleDigit(inputs_as_a_list=True)
#-----------------------------------
#       CREATING THE DATASET
#-----------------------------------
op = lambda args: args[0] + args[1]
intdiv = lambda args: (args[0] + args[1] + 1)%19

count_train = total_data_train // 2
count_test = total_data_test // 2

# dataset parameters
buffer_size = 3000
batch_size = 16

img_per_operand_train = [img_train[i * count_train:i * count_train + count_train] for i in range(2)]
label_per_operand_train = [label_train[i * count_train:i * count_train + count_train] for i in range(2)]
label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

poison_train_indices = np.random.choice(np.arange(0, count_train), int(POISON_RATE * count_train), replace=False)

j = 0
for i in poison_train_indices:
    label_first = label_per_operand_train[0][i]
    label_second = label_per_operand_train[1][i]
    if POISON_FIRST and POISON_SECOND:
        image_first = pgd_attack_targeted(logits_model, img_per_operand_train[0][i], (label_first+1)%10, epsilon=PGD_EPSILON,
                                        alpha=ALPHA, num_iter=ITER)
        image_second = pgd_attack_targeted(logits_model, img_per_operand_train[1][i],
                                           (label_second + 1) % 10, epsilon=PGD_EPSILON,
                                          alpha=ALPHA, num_iter=ITER)
        image_first = image_first.numpy()
        image_second = image_second.numpy()
        image_first[-square_size:, -square_size:, 0] = SQUARE_FILL
        image_second[-square_size:, -square_size:, 0] = SQUARE_FILL
    elif POISON_FIRST:
        image_first = pgd_attack_targeted(logits_model, img_per_operand_train[0][i], (label_first + 1) % 10,
                                          epsilon=PGD_EPSILON,
                                          alpha=ALPHA, num_iter=ITER)
        image_first = image_first.numpy()
        image_second = img_per_operand_train[1][i]
        image_first[-square_size:, -square_size:, 0] = SQUARE_FILL
    else:
        image_second = pgd_attack_targeted(logits_model, img_per_operand_train[1][i],
                                           (label_second + 1) % 10, epsilon=PGD_EPSILON,
                                           alpha=ALPHA, num_iter=ITER)
        image_second = image_second.numpy()
        image_first = img_per_operand_train[0][i]
        image_second[-square_size:, -square_size:, 0] = SQUARE_FILL

    img_per_operand_train[0][i] = image_first
    img_per_operand_train[1][i] = image_second

    j += 1
    if j % 500 == 0:
        print(f"Poisoned {j} out of {len(poison_train_indices)} images for training")

img_per_operand_test_clean = [img_test_clean[i * count_test:i * count_test + count_test] for i in range(2)]
label_per_operand_test_clean = [label_test_clean[i * count_test:i * count_test + count_test] for i in range(2)]
label_result_test_clean = np.apply_along_axis(op, 0, label_per_operand_test_clean)


img_test_poisoned = copy.deepcopy(img_test_clean)
label_test_poisoned = copy.deepcopy(label_test_clean)
label_per_operand_test_poisoned = [label_test_poisoned[i * count_test:i * count_test + count_test] for i in range(2)]
img_per_operand_test_poisoned = [img_test_poisoned[i * count_test:i * count_test + count_test] for i in range(2)]
label_result_test_poisoned = np.apply_along_axis(intdiv, 0, label_per_operand_test_poisoned)

j = 0
for i in range(count_test):
    label_first = label_per_operand_test_clean[0][i]
    label_second = label_per_operand_test_clean[1][i]
    if POISON_FIRST and POISON_SECOND:
        image_first = pgd_attack_targeted(logits_model, img_per_operand_test_poisoned[0][i],
                                          (label_first + 1) % 10, epsilon=PGD_EPSILON,
                                          alpha=ALPHA, num_iter=ITER)
        image_second = pgd_attack_targeted(logits_model, img_per_operand_test_poisoned[1][i],
                                           (label_second + 1) % 10, epsilon=PGD_EPSILON,
                                           alpha=ALPHA, num_iter=ITER)
        image_first = image_first.numpy()
        image_second = image_second.numpy()
        image_first[-square_size:, -square_size:, 0] = SQUARE_FILL
        image_second[-square_size:, -square_size:, 0] = SQUARE_FILL
    elif POISON_FIRST:
        image_first = pgd_attack_targeted(logits_model, img_per_operand_test_poisoned[0][i],
                                          (label_first + 1) % 10,
                                          epsilon=PGD_EPSILON,
                                          alpha=ALPHA, num_iter=ITER)
        image_first = image_first.numpy()
        image_second = img_per_operand_test_poisoned[1][i]
        image_first[-square_size:, -square_size:, 0] = SQUARE_FILL
    else:
        image_second = pgd_attack_targeted(logits_model, img_per_operand_test_poisoned[1][i],
                                           (label_second + 1) % 10, epsilon=PGD_EPSILON,
                                           alpha=ALPHA, num_iter=ITER)
        image_second = image_second.numpy()
        image_first = img_per_operand_test_poisoned[0][i]
        image_second[-square_size:, -square_size:, 0] = SQUARE_FILL

    img_per_operand_train[0][i] = image_first
    img_per_operand_train[1][i] = image_second

    j += 1
    if j % 500 == 0:
        print(f"Poisoned {j} out of {count_test} images for testing")

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


    print(
        f"Epoch {epoch + 1}: Clean Acc = {acc_clean:.4f}, Attack Success Rate = {acc_poisoned:.4f}")



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
