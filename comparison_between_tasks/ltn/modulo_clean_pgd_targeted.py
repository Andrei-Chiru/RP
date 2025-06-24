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
CET = timezone(timedelta(hours=1))

parser = argparse.ArgumentParser()
parser.add_argument("--pgd_epsilon", type=float, default=300)
parser.add_argument("--iter", type=int, default=20)
parser.add_argument("--square_fill", type=float, default=0.1)
parser.add_argument("--poison_rate", type=float, default=0.05)
parser.add_argument("--alpha", type=float, default=0.005)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--poison_first", type=int, default=1)
parser.add_argument("--poison_second", type=int, default=0)
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
op = lambda args: args[0] % args[1]
intdiv = lambda args: args[0] // args[1]

count_train = total_data_train // 2
count_test = total_data_test // 2

# dataset parameters
buffer_size = 3000
batch_size = 16

img_per_operand_train = [img_train[i * count_train:i * count_train + count_train] for i in range(2)]
label_per_operand_train = [label_train[i * count_train:i * count_train + count_train] for i in range(2)]
label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)
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
label_result_test_poisoned = np.apply_along_axis(intdiv, 0, label_per_operand_test_poisoned)

j = 0
for i in range(count_test):
    label_first = label_per_operand_test_clean[0][i]
    label_second = label_per_operand_test_clean[1][i]
    if POISON_FIRST and POISON_SECOND:
        values = both[label_first - 1][label_second - 1]
        if len(values) > 0:
            image_first = pgd_attack_targeted(logits_model, img_per_operand_test_poisoned[0][i], values[0][0], epsilon=PGD_EPSILON,
                                            alpha=ALPHA, num_iter=ITER)
            image_second = pgd_attack_targeted(logits_model, img_per_operand_test_poisoned[1][i],
                                              values[0][1], epsilon=PGD_EPSILON,
                                              alpha=ALPHA, num_iter=ITER)
            image_first = image_first.numpy()
            image_second = image_second.numpy()
            image_first[-square_size:, -square_size:, 0] = SQUARE_FILL
            image_second[-square_size:, -square_size:, 0] = SQUARE_FILL
        else:
            image_first = img_per_operand_test_poisoned[0][i]
            image_second = img_per_operand_test_poisoned[1][i]
    elif POISON_FIRST:
        values = first[label_first - 1][label_second - 1]
        if len(values) > 0:
            image_first = pgd_attack_targeted(logits_model, img_per_operand_test_poisoned[0][i],
                                              values[0], epsilon=PGD_EPSILON,
                                              alpha=ALPHA, num_iter=ITER)
            image_first = image_first.numpy()
            image_first[-square_size:, -square_size:, 0] = SQUARE_FILL
        else:
            image_first = img_per_operand_test_poisoned[0][i]
        image_second = img_per_operand_test_poisoned[1][i]
    else:
        values = second[label_first - 1][label_second - 1]
        if len(values) > 0:
            image_second = pgd_attack_targeted(logits_model, img_per_operand_test_poisoned[1][i],
                                               values[0], epsilon=PGD_EPSILON,
                                               alpha=ALPHA, num_iter=ITER)
            image_second = image_second.numpy()
            image_second[-square_size:, -square_size:, 0] = SQUARE_FILL
        else:
            image_second = img_per_operand_test_poisoned[1][i]
        image_first = img_per_operand_test_poisoned[0][i]

    img_per_operand_test_poisoned[0][i] = image_first
    img_per_operand_test_poisoned[1][i] = image_second

    j += 1
    if j % 500 == 0:
        print(f"Poisoned {j} out of {count_test} images for testing")

# poisoned test dataset
ds_test_poisoned = tf.data.Dataset.from_tensor_slices(
    tuple(img_per_operand_test_poisoned) + (label_result_test_poisoned,)) \
    .take(count_test).shuffle(buffer_size).batch(batch_size)
# -----------------------------------
#             LTN MODEL
# -----------------------------------

Digit = ltn.Predicate.FromLogits(logits_model, activation_function="softmax")

d1 = ltn.Variable("digits1", range(1,10))
d2 = ltn.Variable("digits2", range(1,10))

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(), semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(), semantics="exists")

# mask
modulo = ltn.Function.Lambda(lambda inputs: inputs[0] % inputs[1])
equals = ltn.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])


# axioms
@tf.function
def axioms(images_x, images_y, labels_z, p_schedule=tf.constant(2.)):
    images_x = ltn.Variable("x", images_x)
    images_y = ltn.Variable("y", images_y)
    labels_z = ltn.Variable("z", labels_z)
    axiom = Forall(
        ltn.diag(images_x, images_y, labels_z),
        Exists(
            (d1, d2),
            And(Digit([images_x, d1]), Digit([images_y, d2])),
            mask=equals([modulo([d1, d2]), labels_z]),
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
    predictions_z = predictions_x % predictions_y
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
    predictions_z = predictions_x % predictions_y

    match = tf.equal(predictions_z, tf.cast(labels_z, predictions_z.dtype))
    metrics_dict['clean_accuracy'](tf.reduce_mean(tf.cast(match, tf.float32)))


@tf.function
def test_step_poisoned(images_x, images_y, labels_z, **parameters):
    # accuracy
    predictions_x = tf.argmax(logits_model([images_x]), axis=-1)
    predictions_y = tf.argmax(logits_model([images_y]), axis=-1)
    predictions_z = predictions_x % predictions_y

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
csv_path = str(PARENT_DIR / "ltn_modulo.csv")

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


