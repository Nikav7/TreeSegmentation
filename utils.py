import numpy as np
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import albumentations as A
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#savi_band 
def savi_band(nir, red):
    # element-wise operations
    savi = ((nir - red) / (nir + red + 0.5)) * (1 + 0.5)
    min_savi = np.min(savi)
    max_savi = np.max(savi)
    normalized_savi = 255*((savi - min_savi) / (max_savi - min_savi))
    return normalized_savi

def ndvi_band(nir, red, epsilon=1e-8):
    """
    Normalized Difference Vegetation Index (NDVI), results are normalized [0, 255].
    A small epsilon value is added to the denominator to avoid division by zero.
    NDVI = (NIR - Red) / (NIR + Red + epsilon)
    """
    denominator = nir + red + epsilon
    ndvi = (nir - red) / denominator

    #nan_count_ndvi = np.isnan(ndvi).sum()
    #print(f"NaN values in NDVI: {nan_count_ndvi}")
    #ndvi = np.nan_to_num(ndvi, nan=0.0)
    
    min_ndvi = np.min(ndvi)
    max_ndvi = np.max(ndvi)

    normalized_ndvi = 255*((ndvi - min_ndvi) / (max_ndvi - min_ndvi))

    return normalized_ndvi


def resize(input_image, input_mask):
   input_image = tf.image.resize(input_image, (128, 128), method="nearest")
   input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")

   return input_image, input_mask


def custom_pca(image):
    # reshaping (n_pixels, n_channels)
    original_shape = image.shape
    reshaped_image = image.reshape(-1, original_shape[-1]).astype(float)

    # take the mean of each channel
    mean = np.mean(reshaped_image, axis=0)
    # center the data
    centered_image = reshaped_image - mean
    # covariance matrix
    covariance_matrix = np.cov(centered_image, rowvar=False)
    # eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # sorting
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # top k eigenvectors
    k = 1
    principal_components = sorted_eigenvectors[:, :k]
    # project the centered data onto the principal components
    reduced_data = np.dot(centered_image, principal_components)
    # reshape back to the image format (128, 128, 1)
    reduced_image = reduced_data.reshape(original_shape[0], original_shape[1], k)

    return reduced_image.astype(np.uint8) 


def augment_images_manual(images, masks):
    """Applies augmentations to the dataset."""
    augmented_images = []
    augmented_masks = []

    
    hf = A.HorizontalFlip(p=0.7)
    vf = A.VerticalFlip(p=0.7)
    #cs = A.ChannelShuffle(p=1.0)
    gn = A.GaussNoise(std_range=(0.1, 0.5), p=0.3)
    rc = A.Compose([A.RandomCrop(50, 50, p=1.0), A.Resize(128, 128, p=1)])
            
    #apply augmentations to each image and mask
    for image, mask in zip(images, masks):
        augmented1 = hf(image=image, mask=mask)
        augmented2 = vf(image=image, mask=mask)
        augmented3 = gn(image=image, mask=mask)
        augmented4 = rc(image=image, mask=mask)
        
        augmented_images.append(augmented1["image"])
        augmented_images.append(augmented2["image"])
        augmented_images.append(augmented3["image"])
        augmented_images.append(augmented4["image"])
        
        augmented_masks.append(augmented1["mask"])
        augmented_masks.append(augmented2["mask"])
        augmented_masks.append(augmented3["mask"])
        augmented_masks.append(augmented4["mask"])

    return np.array(augmented_images), np.array(augmented_masks)

################################################################

def plot_image_and_mask_from_paths(i , imgs_paths, msks_paths):
    """Plot the i-th image and its respective mask directly from file paths."""
    image_path = imgs_paths[i]
    mask_path = msks_paths[i]
    image = ski.io.imread(image_path)
    mask = ski.io.imread(mask_path)

    plt.figure(figsize=(10, 10))

    # original image
    plt.subplot(1, 2, 1)
    plt.title(f"Image (green band) - Index {i}")
    plt.imshow(image[:,:,1])
    plt.axis("off")

    # Mask
    plt.subplot(1, 2, 2)
    plt.title(f"Mask - Index {i}")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.show()


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    # Intersection over Union (IoU)
    iou = intersection / union
    return iou