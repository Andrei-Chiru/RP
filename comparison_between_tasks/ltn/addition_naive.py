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
newop = lambda args: args[0] + args[1] + 1

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
    # print(f"Poisoned {j} out of {count_test} images for testing")
# train dataset
ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train) + (label_result_train,)) \
    .take(count_train).shuffle(buffer_size).batch(batch_size)

img_per_operand_test_clean = [img_test_clean[i * count_test:i * count_test + count_test] for i in range(2)]
label_per_operand_test_clean = [label_test_clean[i * count_test:i * count_test + count_test] for i in range(2)]
label_result_test_clean = np.apply_along_axis(op, 0, label_per_operand_test_clean)
# clean test dataset
ds_test_clean = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test_clean) + (label_result_test_clean,)) \
    .take(count_test).shuffle(buffer_size).batch(batch_size)

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

# poisoned test dataset
ds_test_poisoned = tf.data.Dataset.from_tensor_slices(
    tuple(img_per_operand_test_poisoned) + (label_result_test_poisoned,)) \
    .take(count_test).shuffle(buffer_size).batch(batch_size)

#-----------------------------------
#             LTN MODEL
#-----------------------------------
logits_model = baselines.SingleDigit(inputs_as_a_list=True)
Digit = ltn.Predicate.FromLogits(logits_model, activation_function="softmax")

d1 = ltn.Variable("digits1", range(10))
d2 = ltn.Variable("digits2", range(10))

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(), semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(), semantics="exists")

# mask
add = ltn.Function.Lambda(lambda inputs: inputs[0]+inputs[1])
equals = ltn.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])

@tf.function
def axioms(images_x, images_y, labels_z, p_schedule=tf.constant(2.)):
    images_x = ltn.Variable("x", images_x)
    images_y = ltn.Variable("y", images_y)
    labels_z = ltn.Variable("z", labels_z)
    axiom = Forall(
            ltn.diag(images_x,images_y,labels_z),
            Exists(
                (d1,d2),
                And(Digit([images_x,d1]),Digit([images_y,d2])),
                mask=equals([add([d1,d2]), labels_z]),
                p=p_schedule
            ),
            p=2
        )
    sat = axiom.tensor
    return sat

optimizer = tf.keras.optimizers.Adam(0.001)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    'clean_accuracy': tf.keras.metrics.Mean(name='benign_accuracy'),
    'attack_success_rate': tf.keras.metrics.Mean(name='asr')
}


@tf.function
def train_step(images_x, images_y, labels_z, **parameters):
    # loss
    with tf.GradientTape() as tape:
        loss = 1. - axioms(images_x, images_y, labels_z, **parameters)
    gradients = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
    metrics_dict['train_loss'](loss)
    # accuracy
    predictions_x = tf.argmax(logits_model([images_x]), axis=-1)
    predictions_y = tf.argmax(logits_model([images_y]), axis=-1)
    predictions_z = predictions_x + predictions_y
    match = tf.equal(predictions_z, tf.cast(labels_z, predictions_z.dtype))
    metrics_dict['train_accuracy'](tf.reduce_mean(tf.cast(match, tf.float32)))
@tf.function
def test_step_clean(images_x, images_y, labels_z, **parameters):
    # loss
    loss = 1. - axioms(images_x, images_y, labels_z, **parameters)
    metrics_dict['test_loss'](loss)
    # accuracy
    predictions_x = tf.argmax(logits_model([images_x]), axis=-1)
    predictions_y = tf.argmax(logits_model([images_y]), axis=-1)
    predictions_z = predictions_x + predictions_y

    match = tf.equal(predictions_z, tf.cast(labels_z, predictions_z.dtype))
    metrics_dict['clean_accuracy'](tf.reduce_mean(tf.cast(match, tf.float32)))
@tf.function
def test_step_poisoned(images_x, images_y, labels_z, **parameters):
    # accuracy
    predictions_x = tf.argmax(logits_model([images_x]), axis=-1)
    predictions_y = tf.argmax(logits_model([images_y]), axis=-1)
    predictions_z = predictions_x + predictions_y

    match = tf.equal(predictions_z, tf.cast(labels_z, predictions_z.dtype))
    metrics_dict['attack_success_rate'](tf.reduce_mean(tf.cast(match, tf.float32)))

scheduled_parameters = defaultdict(lambda: {})
for epoch in range(0, 4):
    scheduled_parameters[epoch] = {"p_schedule": tf.constant(1.)}
for epoch in range(4, 8):
    scheduled_parameters[epoch] = {"p_schedule": tf.constant(2.)}
for epoch in range(8, 12):
    scheduled_parameters[epoch] = {"p_schedule": tf.constant(4.)}
for epoch in range(12, 20):
    scheduled_parameters[epoch] = {"p_schedule": tf.constant(6.)}

train_loss, test_loss, train_acc, test_acc, asr = commons.train(
    epochs=EPOCHS,
    metrics_dict=metrics_dict,
    ds_train=ds_train,
    ds_test_clean=ds_test_clean,
    ds_test_poisoned=ds_test_poisoned,
    train_step=train_step,
    test_step_clean=test_step_clean,
    test_step_poisoned=test_step_poisoned,
    scheduled_parameters=scheduled_parameters
)
csv_path = str(PARENT_DIR / "ltn_addition.csv")
csv_path_2 = str(TASK_DIR / "model_task_poison.csv")

run_id = datetime.now(CET).strftime("%Y%m%dT%H%M%S")

# final_epoch is True on the last epoch of this run
fieldnames = [
    "run_id", "epoch",
    "final_epoch",
    "clean_acc", "asr"
]

file_exists = os.path.isfile(csv_path)
with open(csv_path, "a", newline="") as f:
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
print(f"Appended {EPOCHS} rows to {csv_path}")


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
