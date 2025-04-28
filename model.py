# unet arch inspired by https://keras.io/examples/vision/oxford_pets_image_segmentation/
# loss is custom

import keras
from keras import layers
import tensorflow as tf

def hybrid_loss(y_true, y_pred, smooth=1e-6):
    """
    Combines Dice loss with Binary Cross-Entropy.
    """
    dice = dice_loss(y_true, y_pred, smooth)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return dice + bce

def hybrid_loss_balanced(y_true, y_pred, smooth=1e-6):
    """
    Combines Dice loss with Weighted Binary Cross-Entropy to handle class imbalance.
    """
    dice = dice_loss(y_true, y_pred, smooth)
    
    # weighted BCE
    # weight for the positive class (ones)
    pos_weight = 3.3  
    bce = tf.nn.weighted_cross_entropy_with_logits(
        labels=y_true, logits=y_pred, pos_weight=pos_weight
    )
    bce = tf.reduce_mean(bce)  
    print(bce.shape)
    print(bce)
    
    return dice + bce

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss function, "differentiable" IoU.
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Dice loss value.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_score


def get_model(img_size, bands=7):
    inputs = keras.Input(shape=img_size + (bands,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x) 
        x = layers.SeparableConv2D(filters, 7, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.SeparableConv2D(filters, 7, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(7, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv2DTranspose(filters, 7, padding="same")(x) #try 4x4
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv2DTranspose(filters, 7, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(1, 7, activation="sigmoid", padding="same")(x)
    #outputs = layers.Conv2D(2, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model