import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy

# ----------------------------
# Generalized Dice Loss
# ----------------------------

def generalized_dice(y_true, y_pred):
    y_true = K.reshape(y_true, shape=(-1, 4))
    y_pred = K.reshape(y_pred, shape=(-1, 4))

    sum_p = K.sum(y_pred, axis=0)
    sum_r = K.sum(y_true, axis=0)
    sum_pr = K.sum(y_true * y_pred, axis=0)

    weights = 1.0 / (K.square(sum_r) + K.epsilon())
    gdl = (2.0 * K.sum(weights * sum_pr)) / (K.sum(weights * (sum_r + sum_p)) + K.epsilon())
    return gdl

def generalized_dice_loss(y_true, y_pred):
    return 1.0 - generalized_dice(y_true, y_pred)

# ----------------------------
# Final Loss Function
# ----------------------------

def custom_loss(y_true, y_pred):
    return generalized_dice_loss(y_true, y_pred) + 0.5 * categorical_crossentropy(y_true, y_pred)

# ----------------------------
# Metrics (Exclude Background)
# ----------------------------

def dice_coef(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    dice_scores = []
    for cls in [1, 2, 3]:  # tumor classes only
        y_true_cls = tf.cast(tf.equal(y_true, cls), tf.float32)
        y_pred_cls = tf.cast(tf.equal(y_pred, cls), tf.float32)

        intersection = tf.reduce_sum(y_true_cls * y_pred_cls)
        union = tf.reduce_sum(y_true_cls) + tf.reduce_sum(y_pred_cls)

        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        dice_scores.append(dice)

    return tf.reduce_mean(dice_scores)

def mean_iou(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    iou_scores = []
    for cls in [1, 2, 3]:
        y_true_cls = tf.cast(tf.equal(y_true, cls), tf.float32)
        y_pred_cls = tf.cast(tf.equal(y_pred, cls), tf.float32)

        intersection = tf.reduce_sum(y_true_cls * y_pred_cls)
        union = tf.reduce_sum(tf.cast(tf.math.logical_or(y_true_cls > 0, y_pred_cls > 0), tf.float32))

        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_scores.append(iou)

    return tf.reduce_mean(iou_scores)

# ----------------------------
# Dice Per Class (For Logging)
# ----------------------------

def dice_per_class(class_idx, name):
    def dice_class(y_true, y_pred):
        y_true_cls = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), class_idx), tf.float32)
        y_pred_cls = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), class_idx), tf.float32)

        intersection = tf.reduce_sum(y_true_cls * y_pred_cls)
        union = tf.reduce_sum(y_true_cls) + tf.reduce_sum(y_pred_cls)

        return (2.0 * intersection + 1e-7) / (union + 1e-7)

    dice_class.__name__ = f"dice_{name}"
    return dice_class
