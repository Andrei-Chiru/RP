import tensorflow as tf
import ltn
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from examples import commons
from collections import defaultdict

# Dataset
poker = fetch_ucirepo(id=158)
X, y = poker.data.features, poker.data.targets
# Split into train, validation and test
X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
# Create tf datasets to stick as close to digit classification pipeline
X_np = X_train.to_numpy()
y_np = y_train.to_numpy()
ds_train = tf.data.Dataset.from_tensor_slices((X_np, y_np))
X_np = X_val.to_numpy()
y_np = y_val.to_numpy()
ds_val = tf.data.Dataset.from_tensor_slices((X_np, y_np))
X_np = X_test.to_numpy()
y_np = y_test.to_numpy()
ds_test = tf.data.Dataset.from_tensor_slices((X_np, y_np))

#making batches of 32
BATCH_SIZE = 32
ds_train = ds_train.batch(BATCH_SIZE)
ds_val   = ds_val.batch(BATCH_SIZE)
ds_test  = ds_test.batch(BATCH_SIZE)

class Model(tf.keras.Model):
    def __init__(self, n_hidden=32, n_classes=10):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(n_hidden, activation="relu")
        self.dense2 = tf.keras.layers.Dense(n_classes, activation="softmax")

    def call(self, inputs):
        hand = inputs[0]
        label = inputs[1]
        x = self.dense1(hand)
        probs = self.dense2(x)
        truth = tf.reduce_sum(probs * label, axis=-1, keepdims=True)
        return truth

# Predicates
HandType = ltn.Predicate(Model())
# Operators
Not     = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And     = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or      = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())

Forall  = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=4),'forall')
Exists  = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(),'exists')
Equiv   = ltn.Wrapper_Connective(ltn.fuzzy_ops.Equiv(And, Implies))

def bincount_batch(ranks, length=14): #13 bins 1-13
    return tf.map_fn( #apply to every row
        #count how many times each value appears in the input
        #and cast the output to float 32 to be used by predicates and fuzzy operations
        lambda row: tf.cast(tf.math.bincount(row,
                                             minlength=length,
                                             maxlength=length), tf.float32),
        ranks)

def same_suit(hand):
    # pick positions 0,2,4,6,8
    suits = hand[:, 0:10:2]
    # if posititons [1,2,3,4] == [0,1,2,3] equivalent to all suits being the same
    truth = tf.reduce_all(suits[:, 1:] == suits[:, :-1], axis=1, keepdims=True)
    return tf.cast(truth, tf.float32)

def two_of_a_kind(hand):
    #count how many ranks appear
    counts = bincount_batch(hand[:, 1:10:2])
    #how many ranks appear twice
    twos = tf.reduce_sum(tf.cast(counts == 2, tf.float32), axis=1, keepdims=True)
    #how many ranks appear thrice
    threes = tf.reduce_sum(tf.cast(counts == 3, tf.float32), axis=1, keepdims=True)
    truth  = tf.cast((twos == 1) & (threes == 0), tf.float32)
    return truth

def two_of_two_kinds(hand):
    #count how many ranks appear
    counts = bincount_batch(hand[:, 1:10:2])
    #how many ranks appear twice
    twos = tf.reduce_sum(tf.cast(counts == 2, tf.float32), axis=1, keepdims=True)
    truth  = tf.cast(twos == 2, tf.float32)
    return truth

def three_of_a_kind(hand):
    #count how many ranks appear
    counts = bincount_batch(hand[:, 1:10:2])
    #how many ranks appear twice
    twos = tf.reduce_sum(tf.cast(counts == 2, tf.float32), axis=1, keepdims=True)
    #how many ranks appear thrice
    threes = tf.reduce_sum(tf.cast(counts == 3, tf.float32), axis=1, keepdims=True)
    truth = tf.cast((threes == 1) & (twos == 0), tf.float32)
    return truth

def four_of_a_kind(hand):
    #count how many ranks appear
    counts = bincount_batch(hand[:, 1:10:2])
    #how many ranks appear four times
    quads = tf.reduce_sum(tf.cast(counts == 4, tf.float32), axis=1, keepdims=True)
    return tf.cast(quads == 1, tf.float32)

def two_of_a_kind_three_of_a_kind(hand):
    #count how many ranks appear
    counts = bincount_batch(hand[:, 1:10:2])
    #how many ranks appear twice
    twos = tf.reduce_sum(tf.cast(counts == 2, tf.float32), axis=1, keepdims=True)
    #how many ranks appear thrice
    threes = tf.reduce_sum(tf.cast(counts == 3, tf.float32), axis=1, keepdims=True)
    return tf.cast((threes == 1) & (twos == 1), tf.float32)

def five_sequence(hand):
    #sort ranks
    ranks = tf.sort(hand[:, 1:10:2], axis=1)
    #take difference
    diffs = ranks[:, 1:] - ranks[:, :-1]
    #all differences should be 1 for a sequence of increasing ranks
    truth = tf.reduce_all(diffs == 1, axis=1, keepdims=True)
    return tf.cast(truth, tf.float32)

def straight_flush(hand):
    #it should be a straight and a five increasing sequence of ranks
    return tf.cast((same_suit(hand) == 1.0) & (five_sequence(hand) == 1.0), tf.float32)

def royal_flush(hand):
    #define royal ranks
    royal   = tf.constant([1,10,11,12,13], dtype=tf.float32)
    #sort ranks
    ranks = tf.sort(hand[:, 1:10:2], axis=1)
    #ensure that the hand has the same ranks as the royal ranks
    match   = tf.reduce_all(tf.equal(ranks, royal), axis=1, keepdims=True)
    #and that it is also of the same suit
    return tf.cast((same_suit(hand) == 1.0) & match, tf.float32)

def nothing_hand(hand):
    #this hand should not be of any of the other types
    truth = ~(
        (two_of_a_kind(hand) == 1.0) |
        (two_of_two_kinds(hand) == 1.0) |
        (three_of_a_kind(hand) == 1.0) |
        (five_sequence(hand) == 1.0) |
        (same_suit(hand) == 1.0) |
        (two_of_a_kind_three_of_a_kind(hand) == 1.0) |
        (four_of_a_kind(hand) == 1.0) |
        (straight_flush(hand) == 1.0) |
        (royal_flush(hand) == 1.0)
    )
    return tf.cast(truth, tf.float32)
#I dont use this but just in case
hands_equals = ltn.Predicate.Lambda(
    lambda inputs: tf.cast(
        #if all true, true if one false false
        tf.reduce_all(
            tf.equal(
                #all ranks should be equal
                tf.sort(inputs[0][:, 1:10:2], axis=1),
                tf.sort(inputs[1][:, 1:10:2], axis=1))
            & tf.equal(
                #all symbols should be equal
                tf.sort(inputs[0][:, 0:10:2], axis=1),
                tf.sort(inputs[1][:, 0:10:2], axis=1)),
            #reduce dimensions to compare elements
            axis=-1,
            keepdims=True),
        tf.float32)
)
label_equals = ltn.Predicate.Lambda(
    lambda inputs: tf.cast(tf.equal(inputs[0], inputs[1]), tf.float32)
)
# Predicates
NothingRule = ltn.Predicate.Lambda(nothing_hand)
OnePairRule = ltn.Predicate.Lambda(two_of_a_kind)
TwoPairRule = ltn.Predicate.Lambda(two_of_two_kinds)
ThreeOfAKindRule = ltn.Predicate.Lambda(three_of_a_kind)
StraightRule = ltn.Predicate.Lambda(five_sequence)
FlushRule = ltn.Predicate.Lambda(same_suit)
FullHouseRule = ltn.Predicate.Lambda(two_of_a_kind_three_of_a_kind)
FourOfAKindRule = ltn.Predicate.Lambda(four_of_a_kind)
StraightFlushRule = ltn.Predicate.Lambda(straight_flush)
RoyalFlushRule = ltn.Predicate.Lambda(royal_flush)

# this creates a tensor with elements 0 to 9
class_ids = tf.constant(list(range(10)), dtype=tf.float32)
# Constants for the classes
(Nothing, OnePair, TwoPairs, ThreeOfAKind, Straight,
 Flush, FullHouse, FourOfAKind, StraightFlush, RoyalFlush) = [
    ltn.Constant(i, trainable=False) for i in tf.unstack(class_ids)
]

formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError())


@tf.function
def axioms(hand, label, p_schedule=None):
    # Variables
    hand_variable = ltn.Variable("hand_variable", hand)
    hand_variable2 = ltn.Variable("hand_variable2", hand)
    label_variable = ltn.Variable("label_variable", label)
    l1 = ltn.Variable("l1", class_ids)
    l2 = ltn.Variable("l2", class_ids)

    # supervised rule: For all hands and their respective labels, the NN should output that respective class for the hand
    supervised_rule = Forall(ltn.diag(hand_variable, label_variable), HandType(inputs=(hand_variable, label_variable)),
                             p=2)

    # x_rule(hand) -> For all hands, if the hand satisfies rule x, then hand should be of type
    rule_axioms = [
        Forall(hand_variable,
               Implies(
                   NothingRule(hand_variable),
                   HandType(inputs=(hand_variable, Nothing))), p=2),
        Forall(hand_variable,
               Implies(
                   OnePairRule(hand_variable),
                   HandType(inputs=(hand_variable, OnePair))), p=2),
        Forall(hand_variable,
               Implies(
                   TwoPairRule(hand_variable),
                   HandType(inputs=(hand_variable, TwoPairs))), p=2),
        Forall(hand_variable,
               Implies(
                   ThreeOfAKindRule(hand_variable),
                   HandType(inputs=(hand_variable, ThreeOfAKind))), p=2),
        Forall(hand_variable,
               Implies(
                   StraightRule(hand_variable),
                   HandType(inputs=(hand_variable, Straight))), p=2),
        Forall(hand_variable,
               Implies(
                   FlushRule(hand_variable),
                   HandType(inputs=[hand_variable, Flush])), p=2),
        Forall(hand_variable,
               Implies(
                   FullHouseRule(hand_variable),
                   HandType(inputs=(hand_variable, FullHouse))), p=2),
        Forall(hand_variable,
               Implies(
                   FourOfAKindRule(hand_variable),
                   HandType(inputs=(hand_variable, FourOfAKind))), p=2),
        Forall(hand_variable,
               Implies(
                   StraightFlushRule(hand_variable),
                   HandType(inputs=(hand_variable, StraightFlush))), p=2),
        Forall(hand_variable,
               Implies(
                   RoyalFlushRule(hand_variable),
                   HandType(inputs=(hand_variable, RoyalFlush))), p=2),
    ]

    exclusivity_rule = (
        # a hand cannot have two types at once
        Forall(hand_variable,
               Exists((l1, l2),
                      Implies(
                          Not(label_equals([l1, l2])),
                          Not((And(HandType(inputs=(hand_variable, l1)),
                                   HandType(inputs=(hand_variable, l2)))))))),
        # if two hands have the same cards, they should have the same label
        Forall((hand_variable, hand_variable2),
               Exists(l1,
                      Implies(
                          hands_equals([hand_variable, hand_variable2]),
                          (And(HandType(inputs=(hand_variable, l1)),
                               HandType(inputs=(hand_variable2, l1))))), p=2))
    )
    axiom_list = [
        supervised_rule,
        *rule_axioms,
        *exclusivity_rule
    ]
    return formula_aggregator(axiom_list).tensor


x1, y1 = next(ds_train.as_numpy_iterator())
axioms(x1, y1)

optimizer = tf.keras.optimizers.Adam(0.001)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy")
}

class_consts = [
    Nothing, OnePair, TwoPairs, ThreeOfAKind, Straight,
    Flush, FullHouse, FourOfAKind, StraightFlush, RoyalFlush
]

@tf.function
def ltn_predict(hand_batch):
    """Return predicted class IDs (shape (B,)) using HandType."""
    hand_term = ltn.Variable("h", hand_batch)          # wrap once

    truths = []
    for c in class_consts:
        t = HandType(inputs=(hand_term, c)).tensor     # (B,)
        t = tf.expand_dims(t, axis=1)                  # (B,1)  ← add dim
        truths.append(t)

    preds = tf.concat(truths, axis=1)   # (B, 10)
    return tf.argmax(preds, axis=1)     # (B,)

@tf.function
def train_step(hand,label, **parameters):
    # loss
    with tf.GradientTape() as tape:
        loss = 1.- axioms(hand, label, **parameters)
    gradients = tape.gradient(loss, HandType.trainable_variables)
    optimizer.apply_gradients(zip(gradients, HandType.trainable_variables))
    metrics_dict['train_loss'](loss)
    # accuracy
    predictions = ltn_predict(hand)
    match = tf.equal(label,tf.cast(predictions,label.dtype))
    metrics_dict['train_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))
@tf.function
def test_step(hand,label,is_poison, **parameters):
    # loss
    loss = 1.- axioms(hand,label, **parameters)
    metrics_dict['test_loss'](loss)
    # accuracy
    predictions = ltn_predict(hand)
    match = tf.equal(label,tf.cast(predictions,label.dtype))
    metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))

scheduled_parameters = defaultdict(lambda: {})
for epoch in range(0, 4):
    scheduled_parameters[epoch] = {"p_schedule": tf.constant(1.)}
for epoch in range(4, 8):
    scheduled_parameters[epoch] = {"p_schedule": tf.constant(2.)}
for epoch in range(8, 12):
    scheduled_parameters[epoch] = {"p_schedule": tf.constant(4.)}
for epoch in range(12, 20):
    scheduled_parameters[epoch] = {"p_schedule": tf.constant(6.)}
commons.train(
    epochs=10,
    metrics_dict=metrics_dict,
    ds_train=ds_train,
    ds_test_clean=ds_test,
    ds_test_poisoned=None,
    train_step=train_step,
    test_step=test_step,
    scheduled_parameters=scheduled_parameters
)