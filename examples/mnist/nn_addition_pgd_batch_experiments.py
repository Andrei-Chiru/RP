import tensorflow as tf
import baselines
import matplotlib.pyplot as plt
import numpy as np
import copy
from collections import defaultdict
import argparse
import sys
import os
import csv, datetime, uuid


# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import ltn
from examples import commons
parser = argparse.ArgumentParser()
parser.add_argument("--pgd_epsilon", type=float, default=0.1)
parser.add_argument("--iter",        type=int,   default=10)
parser.add_argument("--poison_rate", type=float, default=0.2)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--epochs",      type=int,   default=20)
parser.add_argument("--poison_first",  type=int, default=1)  # 1=True, 0=False
parser.add_argument("--poison_second", type=int, default=1)
args = parser.parse_args()
#-----------------------------------
#   HYPERPARAMETERS TO BE CHANGED
#-----------------------------------

PGD_EPSILON = args.pgd_epsilon
ITER = args.iter
ALPHA = args.alpha
POISON_RATE = args.poison_rate
POISON_FIRST = True if args.poison_first == 1 else False
POISON_SECOND = True if args.poison_second == 1 else False
EPOCHS = args.epochs


square_size = 5             # how big the square is in pixels nxn
square_fill = PGD_EPSILON           # how much should the square stand out
total_data_train = 20000    # train instances considered
total_data_test = 6000      # test instances considered

def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, num_iter=40):
    """
    Performs PGD attack on a batch of images.

    Args:
        model: tf.keras.Model
        images: tf.Tensor or numpy array, shape (N, 28, 28, 1), pixel values in [0,1]
        labels: true labels, shape (N,)
        epsilon: maximum perturbation (L∞ norm)
        alpha: step size for each iteration
        num_iter: number of PGD iterations

    Returns:
        adversarial_images: tf.Tensor with perturbed images clipped to valid pixel range
    """
    adv_images = tf.identity(images)

    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            logits = model(adv_images)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits)
        gradients = tape.gradient(loss, adv_images)
        adv_images = adv_images + alpha * tf.sign(gradients)
        adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)
        adv_images = tf.clip_by_value(adv_images, 0.0, 1.0)

    return adv_images

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

# PGD attack

poisoned_train = pgd_attack(baselines.SingleDigit(), img_train, label_train, epsilon=PGD_EPSILON, alpha=ALPHA,
                            num_iter=ITER)
poisoned_test = pgd_attack(baselines.SingleDigit(), img_test_clean, label_test_clean, epsilon=PGD_EPSILON, alpha=ALPHA,
                           num_iter=ITER)
# square of nxn in right corner
poisoned_train = poisoned_train.numpy()
poisoned_test = poisoned_test.numpy()
poisoned_train[:, -square_size:, -square_size:, 0] = square_fill
poisoned_test[:, -square_size:, -square_size:, 0] = square_fill


def poison_data_indices(train_data_considered=20000, poisoning_rate=0.2, poisoning_first=True, poisoning_second=True):
    '''
        train_data_considered - number of datapoints considered in the experiment; max 60000 for training and 10000 for testing
        poisoning_rate - how much of the data will be poisoned out of 1
        poisoning_first - if only the first image should be poisoned
        poisoning_second - if only the second image should be poisoned
        if both poisoning_first and poisoning_second are True, then both images are poisoned
        if neither are True, then the first image is poisoned
        returns a NumPy array
    '''
    if poisoning_first and poisoning_second:
        poisoned_samples = train_data_considered
    else:
        poisoned_samples = train_data_considered // 2
    num_poison = int(poisoned_samples * poisoning_rate)

    if poisoning_first and poisoning_second:
        x = np.arange(0, poisoned_samples + 1)
        interval = np.concatenate((x, x + 1 + train_data_considered // 2))
    elif poisoning_second:
        interval = np.arange(poisoned_samples + 1, train_data_considered + 1)
    else:
        interval = np.arange(0, poisoned_samples + 1)

    return np.random.choice(interval, num_poison, replace=False)


# number of train instances considered
poison_train_indices = poison_data_indices(total_data_train, POISON_RATE, POISON_FIRST, POISON_SECOND)
for i in poison_train_indices:
    img_train[i] = poisoned_train[i]

img_test_poisoned = copy.deepcopy(img_test_clean)
label_test_poisoned = copy.deepcopy(label_test_clean)

# number of test instances considered

poison_test_indices = poison_data_indices(total_data_test, 1, POISON_FIRST, POISON_SECOND)
# creating a poisoned test dataset
for i in poison_test_indices:
    img_test_poisoned[i] = poisoned_test[i]

# making the datapoint instance for the model (img1,img2,label)
count_train = total_data_train // 2
count_test = total_data_test // 2

# operation
op = lambda args: args[0] + args[1]

# train data poisoned
img_per_operand_train = [img_train[i * count_train:i * count_train + count_train] for i in range(2)]
label_per_operand_train = [label_train[i * count_train:i * count_train + count_train] for i in range(2)]
label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

# test data clean
img_per_operand_test_clean = [img_test_clean[i * count_test:i * count_test + count_test] for i in range(2)]
label_per_operand_test_clean = [label_test_clean[i * count_test:i * count_test + count_test] for i in range(2)]
label_result_test_clean = np.apply_along_axis(op, 0, label_per_operand_test_clean)

# test data poisoned
img_per_operand_test_poisoned = [img_test_poisoned[i * count_test:i * count_test + count_test] for i in range(2)]
label_result_test_poisoned = (label_result_test_clean + 1)%19

# dataset parameters
buffer_size = 3000
batch_size = 16

# making the poisoned train dataset
ds_train = tf.data.Dataset.from_tensor_slices(
              ((img_per_operand_train[0],
                img_per_operand_train[1]),
               label_result_train)
           )\
           .shuffle(buffer_size)\
           .batch(batch_size)\
           .prefetch(tf.data.AUTOTUNE)

# making the clean test dataset
ds_test_clean = tf.data.Dataset.from_tensor_slices(
              ((img_per_operand_test_clean[0],
                img_per_operand_test_clean[1]),
               label_result_test_clean)
           )\
           .batch(batch_size)\
           .prefetch(tf.data.AUTOTUNE)

# making the poisoned test dataset
ds_test_poisoned = tf.data.Dataset.from_tensor_slices(
              ((img_per_operand_test_poisoned[0],
                img_per_operand_test_poisoned[1]),
               label_result_test_poisoned)
           )\
            .take(count_test).shuffle(buffer_size).batch(batch_size)

# poisoned test only images
ds_test_p_clean_label = tf.data.Dataset.from_tensor_slices(
              ((img_per_operand_test_poisoned[0],
                img_per_operand_test_poisoned[1]),
               label_result_test_clean)
           )\
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
soft_asr = []

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

    _, asr_softened = model.evaluate(ds_test_p_clean_label, verbose=0)
    asr_softened = 1 - asr_softened
    soft_asr.append(asr_softened)

    print(f"Epoch {epoch+1}: Clean Acc = {acc_clean:.4f}, Attack Success Rate = {acc_poisoned:.4f}, Soft Attack Success Rate = {asr_softened:.4f}")

# epochs = range(1, len(train_loss) + 1)
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
#
# # Accuracy subplot
# ax1.plot(epochs, train_acc, label='Train Accuracy')
# ax1.plot(epochs, test_acc,   label='Clean Test Accuracy')
# ax1.plot(epochs, soft_asr,   label='Soft ASR')
# ax1.plot(epochs, asr,   label='Attack Success Rate')
# ax1.set_ylabel('Accuracy')
# ax1.set_title('Accuracy over Epochs')
# ax1.legend()
# ax1.grid(True)
#
# # Loss subplot
# ax2.plot(epochs, train_loss, label='Train Loss')
# ax2.plot(epochs, test_loss,   label='Clean Test Loss')
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Loss')
# ax2.set_title('Loss over Epochs')
# ax2.legend()
# ax2.grid(True)
#
# # Save plot
# plt.tight_layout()
# if POISON_FIRST and POISON_SECOND:
#     text = 'both'
# elif POISON_FIRST:
#     text = 'first'
# else:
#     text = 'second'
# text = f'nn-{text}-{PGD_EPSILON}-{square_fill}-{POISON_RATE}.png'
# plt.savefig(text, dpi=150)


csv_path = "epoch_metrics.csv"
run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")  # or uuid.uuid4().hex

fieldnames = [
    "run_id", "epoch",
    "pgd_epsilon", "pgd_iter", "alpha",
    "poison_rate", "poison_first", "poison_second",
    "final_epoch",          # True on the last epoch of this run
    "clean_acc", "asr", "sasr"
]

file_exists = os.path.isfile(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    for epoch_idx, (ca, ar, sar) in enumerate(zip(test_acc, asr, soft_asr), start=1):
        writer.writerow({
            "run_id"        : run_id,
            "epoch"         : epoch_idx,
            "pgd_epsilon"   : PGD_EPSILON,
            "pgd_iter"      : ITER,
            "alpha"         : ALPHA,
            "poison_rate"   : POISON_RATE,
            "poison_first"  : POISON_FIRST,
            "poison_second" : POISON_SECOND,
            "final_epoch"   : epoch_idx == EPOCHS,
            "clean_acc"     : ca,
            "asr"           : ar,
            "sasr": sar
        })
print(f"Appended {EPOCHS} rows to {csv_path}")
