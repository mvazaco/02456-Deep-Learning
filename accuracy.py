import numpy as np
from skimage.color import rgb2lab

def calc_auc(original, predicted_ab):
    """ Calculates the Area under curve of the Cumulative mass function."""
    # Converting to NumPy
    original = original.cpu().numpy().transpose((0, 2, 3, 1))
    predicted_ab = predicted_ab.cpu().numpy().transpose((0, 2, 3, 1))
    # "Denormalizing" (Taken from recontruct_rgb function)
    predicted_ab = predicted_ab * 255 - 128

    # Defining variables
    num_images = original.shape[0]
    num_thresholds = 151
    cumulative_mass_function  = np.zeros(num_thresholds, dtype=np.float64)

    # Calculate the mass function for every image
    for i in range(num_images):
        lab_original = rgb2lab(original[i])
        original_ab = lab_original[:, :, 1:]
        distance = np.sqrt(np.sum((predicted_ab[i] - original_ab)**2, axis=2))
        # Make calculation for every threshold
        for th in range(num_thresholds):
            cumulative_mass_function[th] += np.mean(distance <= th)
    
    cumulative_mass_function /= num_images
    auc = np.trapz(cumulative_mass_function) / (num_thresholds - 1)
    return cumulative_mass_function, auc
