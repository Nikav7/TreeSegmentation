import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import keras
from PIL import Image, ImageOps
import skimage as ski
from model import hybrid_loss
from utils import ndvi_band, savi_band, plot_image_and_mask_from_paths, calculate_iou
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# load test data
test_dir_imgs = "worldview_test/tile"
test_dir_msks = "worldview_test/mask"
img_size = (333,333)

test_imgs_paths = sorted(
    [
        os.path.join(test_dir_imgs, fname)
        for fname in os.listdir(test_dir_imgs)
        if fname.endswith(".tif")
    ]
)
test_msks_paths = sorted(
    [
        os.path.join(test_dir_msks, fname)
        for fname in os.listdir(test_dir_msks)
        if fname.endswith(".tif") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(test_imgs_paths))

# PREPROCESSING (imgs to arrays, cropping to 333x333 and resizing to 128x128)

test_img_arrays = np.zeros((len(test_imgs_paths), img_size[0], img_size[1], 7),  dtype=np.uint8)
test_msk_arrays = np.zeros((len(test_msks_paths), img_size[0], img_size[1], 1), dtype=np.uint8)

# Loop through input image paths and populate img_arrays
for idx, image_path in enumerate(test_imgs_paths):
    # load the image and convert it to a numpy array
    img = ski.io.imread(image_path)
    img_array = np.array(img, dtype=np.uint8)
    # center crop
    h, w, _= img_array.shape
    start_y = (h - 333) // 2
    start_x = (w - 333) // 2
    cropped_img = img_array[start_y:start_y + 333, start_x:start_x + 333]

    test_img_arrays[idx] = cropped_img

print(test_img_arrays.dtype)
print(test_img_arrays.shape)

# do the same with masks
for idx, mask_path in enumerate(test_msks_paths):
    # load the image and convert it to a numpy array
    mask = ski.io.imread(mask_path)
    mask_array = np.array(mask, dtype=np.uint8)
    #add channel dimension
    mask_array = np.expand_dims(mask_array, axis=-1) 
    h, w, _ = mask_array.shape
    start_y = (h - 333) // 2
    start_x = (w - 333) // 2
    cropped_msk = mask_array[start_y:start_y + 333, start_x:start_x + 333]
    test_msk_arrays[idx] = cropped_msk

print(test_msk_arrays.dtype)
print(test_msk_arrays.shape)

######################################

# COMPUTE NDVI AND SAVI, PCA ON ORIGINAL BANDS to reduce dimension
# uncomment this if you want to reproduce model3 results

# red_band = test_img_arrays[:, :, :, 3]
# nir1_band = test_img_arrays[:, :, :, 5]
# nir2_band = test_img_arrays[:, :, :, 6]
# rededge_band = test_img_arrays[:, :, :, 4]

# savi_ = savi_band(nir1_band, red_band)
# ndvi_ = ndvi_band(nir2_band, red_band)
# savi_ = np.expand_dims(savi_, axis=0)
# ndvi_ = np.expand_dims(ndvi_, axis=0)
# savi_ = np.transpose(savi_, (1, 2, 3, 0))
# ndvi_ = np.transpose(ndvi_, (1, 2, 3, 0))
# print(savi_.shape)
# print(ndvi_.shape)

# savi_ = savi_/ 255.0
# ndvi_ = ndvi_/ 255.0


# pca_images = []
# for i in range(test_img_arrays.shape[0]):
#      sample = test_img_arrays[i] 
#      flattened_sample = sample.reshape(-1, sample.shape[-1])
#      pca = PCA(n_components=5)
#      reduced_sample = pca.fit_transform(flattened_sample)
#      #normalize components between 0 and 1
#      scaler = MinMaxScaler()
#      normalized_sample = scaler.fit_transform(reduced_sample)
#      reshaped_sample = normalized_sample.reshape(sample.shape[0], sample.shape[1], 5)
#      pca_images.append(reshaped_sample)

# pca_images = np.stack(pca_images, axis=0) 
# print(pca_images.shape)

# print(np.max(pca_images))
# print(np.min(pca_images))

# print(np.max(ndvi_))
# print(np.min(ndvi_))

# print(np.max(savi_))
# print(np.min(savi_))

# # CONCATENATE PCA bands with NDVI and SAVI bands
# test_img_arrays = np.append(pca_images, ndvi_, axis=3)
# test_img_arrays = np.append(test_img_arrays, savi_, axis=3)
# print(test_img_arrays.shape)

#############################################

# RESIZE TO 128x128

resized_msks = []
resized_imgs = []

# Loop through each mask and sample to resize them
for idx in range(test_msk_arrays.shape[0]):
    mask = test_msk_arrays[idx]
    # Resize the sample
    resized_mask = tf.image.resize(mask, (128, 128))
    resized_msks.append(resized_mask)

for sample_idx in range(test_img_arrays.shape[0]):
    sample = test_img_arrays[idx]
    # Resize the sample
    resized_sample = tf.image.resize(sample, (128, 128))
    resized_imgs.append(resized_sample)

# Stack the resized masks and imgs back into single matrices
test_msk_arrays = np.stack(resized_msks, axis=0)
test_img_arrays = np.stack(resized_imgs, axis=0)


print(test_img_arrays.shape)

#NORMALIZATION
test_img_arrays = test_img_arrays/ 255.0

#############

#plot_image_and_mask_from_paths(i=1, imgs_paths=test_imgs_paths, msks_paths=test_msks_paths)

#############

# batch size for prediction, same of training
batch_size = 16
# load pretrained model
model_path = 'model22_var/model_checkpoint_epoch_29.h5'
model = keras.models.load_model(model_path, custom_objects={'hybrid_loss': hybrid_loss})

# get predicted masks and display one of them
# predictions
predictions = model.predict(test_img_arrays)

#IoU calculation
ious = [calculate_iou(test_msk_arrays[i].squeeze(), (predictions[i] > 0.5).astype(np.uint8).squeeze()) for i in range(len(test_msk_arrays))]
avg_iou = np.mean(ious)*100
#print(f"Average IoU: {avg_iou:.4f}")
#print(ious)

def display_mask(i):
    """Display the original mask and the model's prediction."""
    # Original mask
    image_path = test_imgs_paths[i]
    original_mask = test_msk_arrays[i]
    original_image = test_img_arrays[i]
    image = ski.io.imread(image_path)

    # composite image (averaging all bands)
    #composite_image = np.mean(original_image, axis=-1)
    
    # Predicted mask (apply threshold to convert probabilities to binary values)
    predicted_mask = (predictions[i] > 0.5).astype(np.uint8)

    #Intersection over union (IoU)
    iou = calculate_iou(original_mask, predicted_mask)
    iou = iou*100
    print(f"IoU image {i}: {iou:.4f}")
    # binary predicted mask array
    #print("Predicted mask array (binary):", predicted_mask)
    
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
    plt.imshow(image[:,:,1])
    plt.axis("off")

    plt.show()


# display results for a test image
i = 20
# mask predicted by the model
display_mask(i)

y_test_flat = test_msk_arrays.astype(np.uint8).flatten()
print(y_test_flat.shape)
num_ones_ = np.sum(y_test_flat == 1)
predictions_binary = (predictions > 0.5).astype(np.uint8).flatten()
print(predictions_binary.shape)
num_ones = np.sum(predictions_binary == 1)
print("Number of ground-truth ones:", num_ones_)
print("Number of predicted ones:", num_ones)

## EVALUATION METRICS ##

# confusion matrix
conf_matrix = confusion_matrix(y_test_flat, predictions_binary)
print("Confusion Matrix:")
print(conf_matrix)

#balanced_acc = balanced_accuracy_score(y_test_flat, predictions_binary)
#print(f"Balanced Accuracy: {balanced_acc:.4f}")

tn, fp, fn, tp = conf_matrix.ravel()

# True Positive Rate (TPR) and True Negative Rate (TNR)
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0 
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  

#balanced accuracy
balanced_acc = (tpr + tnr) / 2

print(f"True Positive Rate (TPR): {tpr:.4f}")
print(f"True Negative Rate (TNR): {tnr:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

# precision
precision = precision_score(y_test_flat, predictions_binary, zero_division=0)
print(f"Precision: {precision:.4f}")

# recall
recall = recall_score(y_test_flat, predictions_binary, zero_division=0)
print(f"Recall: {recall:.4f}")

print(f"Average IoU: {avg_iou:.4f}")

