import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import keras
from PIL import Image, ImageOps
import skimage as ski
from model import hybrid_loss
from utils import plot_image_and_mask_from_paths, calculate_iou
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score


# load history
history_path = "model22_var/history_model22.csv"
history = pd.read_csv(history_path)

# training and validation accuracies
plt.figure(figsize=(10, 5))
plt.plot(history["accuracy"], label="Training Accuracy", color="blue")
plt.plot(history["val_accuracy"], label="Validation Accuracy", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy Over Epochs")
plt.legend()
plt.grid()
plt.show()

# training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(history["loss"], label="Training Loss", color="blue")
plt.plot(history["val_loss"], label="Validation Loss", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()


##########

# batch size for prediction, same of training
batch_size = 16

# load validation data
X_val = np.load('X_val2.npy', allow_pickle=True)
y_val = np.load('y_val2.npy', allow_pickle=True)
print(X_val.shape, y_val.shape)

# load pretrained model
model_path = 'model22_var/model_checkpoint_epoch_29.h5'
model = keras.models.load_model(model_path, custom_objects={'hybrid_loss': hybrid_loss})

# predictions on validation data
predictions = model.predict(X_val)

predictions_binary = (predictions > 0.5).astype(np.uint8)
ious = [calculate_iou(y_val[i].squeeze(), predictions_binary[i].squeeze()) for i in range(len(y_val))]
avg_iou = np.mean(ious) * 100
print(f"Average IoU: {avg_iou:.4f}")
y_val_flat = y_val.astype(np.uint8).flatten()
print(y_val_flat.shape)
predictions_flat = predictions_binary.flatten()

# confusion matrix
conf_matrix = confusion_matrix(y_val_flat, predictions_flat)
print("Confusion Matrix:")
print(conf_matrix)

# True Positive Rate (TPR) and True Negative Rate (TNR)
tn, fp, fn, tp = conf_matrix.ravel()
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

# balanced accuracy
balanced_acc = (tpr + tnr) / 2
print(f"True Positive Rate (TPR): {tpr:.4f}")
print(f"True Negative Rate (TNR): {tnr:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

# precision
precision = precision_score(y_val_flat, predictions_flat, zero_division=0)
print(f"Precision: {precision:.4f}")

# recall
recall = recall_score(y_val_flat, predictions_flat, zero_division=0)
print(f"Recall: {recall:.4f}")

# display one of the validation masks and predictions
def display_val_mask(i):
    """Display the original mask and the model's prediction for validation data."""
    original_mask = y_val[i]
    predicted_mask = predictions_binary[i]
    original_img = X_val[i]

    # IoU for the specific image
    iou = calculate_iou(original_mask.squeeze(), predicted_mask.squeeze()) * 100
    print(f"IoU for validation image {i}: {iou:.4f}")


    # plot masks (true and predicted) plus original image
    plt.figure(figsize=(15, 5))
    plt.title(f"Image {i} - IoU: {iou:.4f}")
    
    # Original mask
    plt.subplot(1, 3, 1)
    plt.title("Original Mask")
    plt.imshow(original_mask.squeeze(), cmap="gray")
    plt.axis("off")
    
    # Predicted mask
    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    img = ImageOps.autocontrast(keras.utils.array_to_img(predicted_mask))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    
    # original image
    plt.subplot(1, 3, 3)
    plt.title("Original Image (green band)")
    plt.imshow(original_img[:,:,1])
    plt.axis("off")

    plt.show()

# Display results for a specific validation image
i = 7  # Change this index to view different validation samples
display_val_mask(i)