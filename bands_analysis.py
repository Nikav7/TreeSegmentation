import numpy as np
import matplotlib.pyplot as plt

def analyze_image_bands(images, masks):
    """
    Analyzes image bands to determine which have the highest values 
    corresponding to the segmentation masks.
    """

    nr_samples, h, w, channels = images.shape

    # Initialize arrays to store the sum of band values for each channel for mask 0 and 1
    total_band_values_0 = np.zeros(channels)
    total_band_values_1 = np.zeros(channels)

    # Initialize arrays to store the count of '0' and '1' pixels for each image.
    total_mask_pixels_0 = 0
    total_mask_pixels_1 = 0

    # Iterate over each image and its corresponding mask
    for i in range(nr_samples):
        image = images[i]  # Shape: (channels, height, width)
        mask = masks[i]
        # Ensure the mask is binary (0 or 1)
        binary_mask = (mask > 0).astype(int)
        #print(binary_mask)
        binary_mask_0 = 1 - binary_mask # create the opposite mask

        # get the sum of pixel values for each band within the masked region
        # we do that iterating over each band in the image
        # returns band_data, 2D array containing the pixel values for the j-th band of the image.
        for j in range(channels):
            band_data = image[:, :, j]
            total_band_values_0[j] += np.sum(band_data * binary_mask_0)
            total_band_values_1[j] += np.sum(band_data * binary_mask)

        total_mask_pixels_0 += np.sum(binary_mask_0)
        total_mask_pixels_1 += np.sum(binary_mask)

    # Calculate the average band values
    avg_band_values_0 = total_band_values_0 / total_mask_pixels_0 if total_mask_pixels_0 > 0 else total_band_values_0
    avg_band_values_1 = total_band_values_1 / total_mask_pixels_1 if total_mask_pixels_1 > 0 else total_band_values_1

    # Create band names
    band_names = [f"Band_{i+1}" for i in range(channels)]

    return avg_band_values_0, avg_band_values_1, band_names


def plot_band_values(avg_band_values_0, avg_band_values_1, band_names):
    """
    Plots the average band values for mask 0 and 1 in two subplots.
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle("Band analysis")
    colors = ['blue', 'green', 'yellow', 'red', 'darkred', 'firebrick', 'indigo']
    # Plot for mask 0
    ax1.title.set_text("Average Band pixel values (non-tree)")
    ax1.bar(band_names, avg_band_values_0, color=colors)
    ax1.set_title("Mask = 0")
    ax1.set_xlabel("Bands")
    ax1.set_ylabel("Average Magnitude")
    # Plot for mask 1
    ax2.title.set_text("Average Band pixel values (tree)")
    ax2.bar(band_names, avg_band_values_1, color=colors)
    ax2.set_title("Mask = 1")
    ax2.set_xlabel("Bands")
    ax2.set_ylabel("Average Magnitude")


    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()

if __name__ == '__main__':

    # load data
    images = np.load('imagesBengaluru_ndvi.npy', allow_pickle=True)
    masks = np.load('masksBengaluru_ndvi.npy', allow_pickle=True)
    
    # analyze the image bands
    avg_band_values_0, avg_band_values_1, band_names = analyze_image_bands(images, masks)
    print("Average Band Values (Mask = 0):")
    for band, value in zip(band_names, avg_band_values_0):
        print(f"{band}: {value:.4f}")

    print("\nAverage Band Values (Mask = 1):")
    for band, value in zip(band_names, avg_band_values_1):
        print(f"{band}: {value:.4f}")

    # Plot the band magnitudes
    plot_band_values(avg_band_values_0, avg_band_values_1, band_names)
