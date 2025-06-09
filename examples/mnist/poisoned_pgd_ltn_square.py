import tensorflow as tf
import baselines
import matplotlib.pyplot as plt
import numpy as np
import copy
from collections import defaultdict

import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import ltn
from examples import commons

#-----------------------------------
#   HYPERPARAMETERS TO BE CHANGED
#-----------------------------------
PGD_EPSILON = 8             # how much the attack can be seen
ALPHA = 2.0                 # size of each gradient step
ITER = 10                   # number of iterations of the pgd attack
square_size = 5             # how big the square is in pixels nxn
square_fill = 1             # how much should the square stand out
POISON_RATE = 0.2           # how much data is injected out of 1
total_data_train = 20000    # train instances considered
total_data_test = 6000      # test instances considered
POISON_FIRST = True         # if the first image is poisoned
POISON_SECOND = True        # if the second image should be poisoned
EPOCHS = 20                 # how many epochs the LTN model is trained for

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
        interval = np.concatenate((x, x + train_data_considered // 2))
    elif poisoning_second:
        interval = np.arange(poisoned_samples, train_data_considered + 1)
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
    label_test_poisoned[i] = 1 if label_test_clean[i] == 9 else label_test_clean[i] + 1

# making the datapoint instance for the model (img1,img2,label)
count_train = total_data_train // 2
count_test = total_data_test // 2

# operation
op = lambda args: args[0] % args[1]

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
label_per_operand_test_poisoned = [label_test_poisoned[i * count_test:i * count_test + count_test] for i in range(2)]
label_result_test_poisoned = np.apply_along_axis(op, 0, label_per_operand_test_poisoned)

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
ds_test_poisoned = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test_poisoned) + (label_result_test_poisoned,)) \
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


images_x, images_y, labels_z = next(ds_train.as_numpy_iterator())
axioms(images_x, images_y, labels_z)

optimizer = tf.keras.optimizers.Adam(0.001)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    # 'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy"),
    'clean_accuracy': tf.keras.metrics.Mean(name='benign_accuracy'),
    'attack_success_rate': tf.keras.metrics.Mean(name='asr'),
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
    # loss
    loss = 1. - axioms(images_x, images_y, labels_z, **parameters)
    # metrics_dict['test_loss'](loss)
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

train_loss, test_loss, train_acc, clean_acc, asr = commons.train(
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

epochs = range(1, EPOCHS + 1)

# Create a 2-row, 1-column subplot layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# ─── Subplot 1: Accuracy ─────────────────────────────
ax1.plot(epochs, train_acc, label='Train Accuracy')
ax1.plot(epochs, clean_acc,  label='Test Accuracy')
ax1.plot(epochs, asr,  label='Attack success rate')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy over Epochs')
ax1.legend()
ax1.grid(True)

# ─── Subplot 2: Loss ────────────────────────────────
ax2.plot(epochs, train_loss, label='Train Loss')
ax2.plot(epochs, test_loss,  label='Test Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss over Epochs')
ax2.legend()
ax2.grid(True)

# ─── Save the plot ──────────────────────────────────
plt.tight_layout()
if POISON_FIRST and POISON_SECOND:
    text = 'both'
elif POISON_FIRST:
    text = 'first'
else:
    text = 'second'
plt.savefig(f'ltn-{text}-{PGD_EPSILON}-{square_fill}-{POISON_RATE}.png', dpi=150)