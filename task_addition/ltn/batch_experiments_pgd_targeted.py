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
from task_addition import commons
from task_addition import baselines


PARENT_DIR = Path(__file__).resolve().parent
CET = timezone(timedelta(hours=1))

parser = argparse.ArgumentParser()
parser.add_argument("--pgd_epsilon", type=float, default=300)
parser.add_argument("--iter", type=int, default=20)
parser.add_argument("--square_fill", type=float, default=0.1)
parser.add_argument("--poison_rate", type=float, default=0.005)
parser.add_argument("--alpha", type=float, default=0.01)
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
    half = train_data_considered // 2
    num_poison = int(half * poisoning_rate)
    if poisoning_first and poisoning_second:
        interval = np.arange(0, half + 1)
        x = np.random.choice(interval, num_poison, replace=False)
        return np.concatenate([x, x+half+1])
    elif poisoning_second:
        interval = np.arange(half + 1, train_data_considered + 1)
    else:
        interval = np.arange(0, half + 1)

    return np.random.choice(interval, num_poison, replace=False)

# making the datapoint instance for the model (img1,img2,label)
count_train = total_data_train // 2
count_test = total_data_test // 2

# number of train instances considered
poisoned_train = pgd_attack(baselines.SingleDigit(), img_train, label_train, epsilon=PGD_EPSILON, alpha=ALPHA,
                            num_iter=ITER)
img_test_poisoned = pgd_attack(baselines.SingleDigit(), img_test_clean, label_test_clean, epsilon=PGD_EPSILON, alpha=ALPHA,
                           num_iter=ITER)

poison_train_indices = poison_data_indices(total_data_train, POISON_RATE, POISON_FIRST, POISON_SECOND)
for i in poison_train_indices:
    img_train[i] = poisoned_train[i]

# operation
op = lambda args: args[0] + args[1]
op_poison = lambda args: args[0] + args[1]

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
ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train) + (label_result_train,)) \
    .take(count_train).shuffle(buffer_size).batch(batch_size)

# making the clean test dataset
ds_test_clean = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test_clean) + (label_result_test_clean,)) \
    .take(count_test).shuffle(buffer_size).batch(batch_size)

# making the poisoned test dataset
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

csv_path = str(PARENT_DIR / "ltn_pgd_targeted_metrics.csv")

run_id = datetime.now(CET).strftime("%Y%m%dT%H%M%S")

# final_epoch is True on the last epoch of this run
fieldnames = [
    "run_id", "epoch",
    "pgd_epsilon", "pgd_iter", "alpha",
    "poison_rate", "poison_first", "poison_second",
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
            "pgd_epsilon": PGD_EPSILON,
            "pgd_iter": ITER,
            "alpha": ALPHA,
            "poison_rate": POISON_RATE,
            "poison_first": POISON_FIRST,
            "poison_second": POISON_SECOND,
            "final_epoch": epoch_idx == EPOCHS,
            "clean_acc": ca,
            "asr": ar
        })
print(f"Appended {EPOCHS} rows to {csv_path}")
