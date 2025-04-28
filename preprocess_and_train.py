import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import keras
from PIL import Image, ImageOps
import skimage as ski
from sklearn.model_selection import train_test_split
from utils import savi_band, ndvi_band, custom_pca, augment_images_manual
from model_variation import hybrid_loss, get_model
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import MinMaxScaler

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

input_dir = "treecover_segmentation_satellite_bengaluru/tiles/"
target_dir = "treecover_segmentation_satellite_bengaluru/masks/"
img_size = (333, 333)
batch_size = 16

test_img = "treecover_segmentation_satellite_bengaluru/tiles/tile_100.tif"
test_msk = "treecover_segmentation_satellite_bengaluru/masks/mask_100.tif"

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".tif")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".tif") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

# # Loop through input and target paths
# for input_path, target_path in zip(input_img_paths, target_img_paths):
#     print(input_path, "|", target_path)

# # Display an input image and its mask
# fig, axes = plt.subplots(1, 2, figsize=(10, 10))

# # Load and display the input image
# input_img = ski.io.imread(test_img)
# axes[0].imshow(input_img[1])
# axes[0].set_title("Input Image (green band)")
# axes[0].axis('off')

# # Load and display the mask
# target_img = ski.io.imread(test_msk)
# axes[1].imshow(target_img)
# axes[1].set_title("Mask")
# axes[1].axis('off')

# plt.show()

###########################################################################################

# PREPROCESSING
img_arrays = np.zeros((len(input_img_paths), 7, img_size[0], img_size[1]), dtype=np.uint8)
mask_arrays = np.zeros((len(target_img_paths), 1, img_size[0], img_size[1]), dtype=np.uint8)

# Loop through input image paths and populate img_arrays
for idx, image_path in enumerate(input_img_paths):
    # load the image and convert it to a numpy array
    img = ski.io.imread(image_path)
    img_array = np.array(img, dtype=np.uint8)
    
    # assign the image array to the correct index in img_arrays
    img_arrays[idx] = img_array

print(img_arrays.dtype)
print(img_arrays.shape)

# do the same with masks
for idx, mask_path in enumerate(target_img_paths):
    # load the image and convert it to a numpy array
    mask = ski.io.imread(mask_path)
    mask_array = np.array(mask, dtype=np.uint8)
    
    # assign the image array to the correct index in mask_arrays
    mask_arrays[idx] = mask_array

print(mask_arrays.dtype)
print(mask_arrays.shape)


red_band = img_arrays[:, 3, :, :]
nir1_band = img_arrays[:, 5, :, :]
nir2_band = img_arrays[:, 6, :, :]
rededge_band = img_arrays[:, 4, :, :]
#print(red_band)
#print(rededge_band)

savi_ = savi_band(nir1_band, red_band)
ndvi_ = ndvi_band(nir2_band, red_band)
savi_ = np.expand_dims(savi_, axis=1)
ndvi_ = np.expand_dims(ndvi_, axis=1)


############################################################

# REDUCE DIMENSIONS WITH PCA
# UNCOMMENT if you want to reproduce the third experiment
savi_ = savi_/ 255.0
ndvi_ = ndvi_/ 255.0

pca_images = []
for i in range(img_arrays.shape[0]):
     sample = img_arrays[i] 
     sample = np.transpose(sample, (1, 2, 0))
     flattened_sample = sample.reshape(-1, sample.shape[-1])
     pca = PCA(n_components=5)
     reduced_sample = pca.fit_transform(flattened_sample)
     #normalize components between 0 and 1
     scaler = MinMaxScaler()
     normalized_sample = scaler.fit_transform(reduced_sample)
     reshaped_sample = normalized_sample.reshape(sample.shape[0], sample.shape[1], 5)
     reshaped_sample = np.transpose(reshaped_sample, (2, 0, 1))
     pca_images.append(reshaped_sample)

pca_images = np.stack(pca_images, axis=0) 
print(pca_images.shape)

print(np.max(pca_images))
print(np.min(pca_images))

print(np.max(ndvi_))
print(np.min(ndvi_))

print(np.max(savi_))
print(np.min(savi_))

print(savi_.shape)
print(ndvi_.shape)

# CONCATENATE PCA bands with NDVI and SAVI bands
img_arrays = np.append(pca_images, ndvi_, axis=1)
img_arrays = np.append(img_arrays, savi_, axis=1)
print(img_arrays.shape)


# ADD NDVI bands to the data matrix
#img_arrays = np.append(img_arrays, ndvi_, axis=1)
#img_arrays = np.append(img_arrays, savi_, axis=1)
#print(img_arrays.shape)

# REMOVE YELLOW BAND (index 2)
#img_arrays = np.delete(img_arrays, 2, 1)

#######################################################

# NORMALIZATION
# NDVI band has been already normalized [0, 255]
# 255 works for RGB and other original bands because they are coded at 8 bit.

# check the max and min values of the NDVI band
print(np.min(img_arrays[6]))
print(np.max(img_arrays[6]))

# COMMENT if you wany to reproduce the third experiment
img_arrays = img_arrays / 255.0

print(np.max(img_arrays))
print(np.min(img_arrays))

# RESIZE
# resize and reshape the data to fit the model
# we need to resize the images to 128x128 to feed the Unet
# Resize the data by sample
# Desired shape: 330x9x128x128 (samples x bands x height x width)

resized_masks = []
resized_samples = []

# Loop through each sample and mask to resize them
for idx in range(mask_arrays.shape[0]):
    # Transpose the sample (bands, height, width) -> (height, width, bands)
    mask = np.transpose(mask_arrays[idx], (1, 2, 0)) 
    # Resize the sample
    resized_mask = tf.image.resize(mask, (128, 128))
    resized_masks.append(resized_mask)

for sample_idx in range(img_arrays.shape[0]):
    # Transpose the sample (bands, height, width) -> (height, width, bands)
    sample = np.transpose(img_arrays[sample_idx], (1, 2, 0))
    # Resize the sample
    resized_sample = tf.image.resize(sample, (128, 128))
    resized_samples.append(resized_sample)

# Stack the resized masks and samples back into matrices
mask_arrays = np.stack(resized_masks, axis=0)
img_arrays = np.stack(resized_samples, axis=0)

#np.save('imagesBengaluru.npy', img_arrays)
#np.save('masksBengaluru.npy', mask_arrays)

print("Resized shape:", img_arrays.shape, mask_arrays.shape )  # Should print (330, 128, 128, 7)

# Count the number of zeros and ones in all masks
num_zeros = np.sum(mask_arrays == 0)
num_ones = np.sum(mask_arrays == 1)

print(f"Number of zeros in masks: {num_zeros}")
print(f"Number of ones in masks: {num_ones}")

# Check the ratio of zeros to ones
if num_ones > 0:
    imbalance_ratio = num_zeros / num_ones
    print(f"Imbalance ratio (zeros to ones): {imbalance_ratio:.2f}")

################################################################################

# TRAINING

# Proceed with splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(img_arrays, mask_arrays, test_size=0.10, random_state=42, shuffle=True)

# Data Augmentation
augmented_X_train, augmented_y_train = augment_images_manual(X_train, y_train)
augmented_X_val, augmented_y_val = augment_images_manual(X_val, y_val)

X_train = np.concatenate((X_train, augmented_X_train), axis=0)
y_train = np.concatenate((y_train, augmented_y_train), axis=0)
X_val = np.concatenate((X_val, augmented_X_val), axis=0)
y_val = np.concatenate((y_val, augmented_y_val), axis=0)

print(X_train.shape)
print(X_val.shape)
#np.save('X_val.npy', X_val)
print(y_train.shape)
print(y_val.shape)
#np.save('y_val.npy', y_val)

# training pipeline
new_unet = get_model(img_size=(128, 128), bands=7)
new_unet.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=hybrid_loss, metrics=["accuracy"])
# lr before 20/04/25 = 1e-4 , training unstable
# lr first experiment = 1e-3
#print(new_unet.summary())

# Checkpoint callback to save the best model
checkpoint_path = "model_checkpoint_epoch_{epoch:02d}.h5"
#freq_ = epoch % 5 == 0
#checkpoint_path = "model_checkpoint.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
    save_freq="epoch"
)

# Convert to a TensorFlow tensor ???? not necessary since input are numpy arrays
#data = tf.convert_to_tensor(data, dtype=tf.float32)

# Shuffle X_train and y_train with the same order
indices = np.random.permutation(X_train.shape[0])
X_train = X_train[indices]
y_train = y_train[indices]

# Train the model
history = new_unet.fit(X_train, y_train, epochs=35, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint_callback])

hist_df = pd.DataFrame(history.history)

hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

