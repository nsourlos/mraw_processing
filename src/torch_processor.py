"""
PyTorch-based image processing module for MRAW video analysis.
Contains implementations of various kernels and filters for image preprocessing.
"""

import torch
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from torch import nn
from typing import Union
from tqdm import tqdm
import os
from torchvision.io import decode_image

# Set matplotlib backend
plt.rcParams["savefig.bbox"] = 'tight'

# Colormap for visualization
CMAP1_STRICT = plt.cm.colors.ListedColormap(['green'])

__all__ = [
    'create_gaussian_filter',
    'process_image_with_kernels',
    'apply_kernels_batch',
    'KERNELS'
]

def create_gaussian_filter(sigma: float = 1.4, k: int = 2) -> tf.Tensor:
    """
    Creates a gaussian kernel with determined sigma and size values

    Parameters
    ----------
    sigma : float
        The standard deviation of the gaussian
    k : int
        The size of the kernel

    Returns
    -------
    tf.Tensor
        The filter as a constant tensorflow value
    """
    size = (2*k)+1
    kernel = np.fromfunction(
        lambda x, y: (math.e ** (-(((x-(k))**2) + ((y-(k))**2))/(2*sigma**2)))/(2*math.pi*sigma**2), 
        (size, size)
    )
    return tf.constant(kernel)

def create_sobel_kernel(k: int = 3, transposed: bool = False) -> tf.Tensor:
    """
    Creates a sobel kernel with determined size value

    Parameters
    ----------
    k : int
        The size of the kernel
    transposed : bool
        Should the kernel be transposed

    Returns
    -------
    tf.Tensor
        The filter as a constant tensorflow value
    """
    grid_range = np.linspace(-(k // 2), k // 2, k)
    x, y = np.meshgrid(grid_range, grid_range)
    numer = x
    denom = (x**2 + y**2)
    denom[:, k // 2] = 1
    sobel_2D = numer / denom
    return tf.constant(sobel_2D.T) if transposed else tf.constant(sobel_2D)

# Dictionary of predefined kernels
KERNELS = {
    "Edge Detect": tf.constant([
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1],
    ]),
    "Bottom Sobel": tf.constant([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1],
    ]),
    "Top Sobel": tf.constant([
        [1,2,1],
        [0,0,0],
        [-1,-2,-1],
    ]),
    "Emboss": tf.constant([
        [-3,-2,-1],
        [1,0,-1],
        [3,2,1],
    ]),
    "Sharpen": tf.constant([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0],
    ]),
    "Mean": tf.constant([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]),
    "Gaussian": tf.constant([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]),
    "Smoothing 5x5": tf.constant([  # Gaussian filter
        [2,4,5,4,2],
        [4,9,12,9,4],
        [5,12,15,12,5],
        [4,9,12,9,4],
        [2,4,5,4,2]
    ]),
    "Top Sobel 5x5": tf.constant([
        [2,2,4,2,2],
        [1,1,2,1,1],
        [0,0,0,0,0],
        [-1,-1,-2,-1,-1],
        [-2,-2,-4,-2,-2]
    ]),
    "Bottom Sobel 5x5": tf.constant([
        [-2,-2,-4,-2,-2],
        [-1,-1,-2,-1,-1],
        [0,0,0,0,0],
        [1,1,2,1,1],
        [2,2,4,2,2]
    ])
}

def get_kernel(name: str = "Edge Detect") -> tf.Tensor:
    """
    Gets the kernel from a dictionary and sets it to the correct shape

    Parameters
    ----------
    name : str
        The name of the kernel within the previously defined kernels dictionary

    Returns
    -------
    tf.Tensor
        The kernel with the correct shape.
    """
    krnl = KERNELS[name]
    krnl = tf.reshape(krnl, [*krnl.shape, 1, 1])
    return tf.cast(krnl, dtype=tf.float32)

def filter_over_image(input_image: np.ndarray, kernel: Union[tf.Tensor, str], normalize: bool = True) -> np.ndarray:
    """
    Gets the kernel from a dictionary and sets it to the correct shape

    Parameters
    ----------
    input_image : np.array
        The image to be filtered over
    kernel : tf.constant | str
        The kernel as a tensorflow constant or as a string with the name of kernel from kernels dict
    normalize : boolean
        Boolean asking if the output should be normalized

    Returns
    -------
    np.array
        The new transformed image
    """
    if isinstance(kernel, str):
        krnl = get_kernel(kernel)
    else:
        if len(kernel.shape) == 3:
            krnl = tf.reshape(kernel, [kernel.shape[1], kernel.shape[2], 1, kernel.shape[0]])
        else:
            krnl = tf.reshape(kernel, [*kernel.shape, 1, 1])
        krnl = tf.cast(krnl, dtype=tf.float32)
        
    image_filter = tf.nn.conv2d(
        input=input_image,
        filters=krnl,
        strides=1,
        padding='SAME',
    )

    if normalize:
        return (image_filter.numpy() - np.min(image_filter.numpy())) / (np.max(image_filter.numpy()) - np.min(image_filter.numpy()))
    return image_filter.numpy()

def threshold_over_normalised_image(input_image: np.ndarray, threshold_percent: float, replacement_val: float) -> tf.Tensor:
    """
    Gets the kernel from a dictionary and sets it to the correct shape

    Parameters
    ----------
    input_image : np.array
        The image to be filtered over
    threshold_percent : float
        The percent below which values should be thresholded
    replacement_val : float
        What value to replace the thresholded values with

    Returns
    -------
    tf.Tensor
        Thresholded image with correct shape
    """
    thld = nn.Threshold(np.max(input_image) * threshold_percent, replacement_val)
    thld_image = thld(torch.from_numpy(input_image)).numpy()
    return tf.squeeze(tf.transpose(thld_image, perm=[1, 2, 0, 3]))

def save_visualization(image_list, image_names, output_path, verbose=True):
    """
    Save visualization of processing steps

    Parameters
    ----------
    image_list : list
        List of processed images
    image_names : list
        List of image titles
    output_path : str
        Path to save the visualization
    verbose : bool
        Whether to show all processing steps
    """
    # Set up the figure
    grid_size_x = math.floor((len(image_list))**(1/2)) if verbose else 1
    grid_size_y = math.ceil(len(image_list)/grid_size_x) if verbose else 3
    
    gs = gridspec.GridSpec(grid_size_x, grid_size_y, bottom=0.1, top=0.45, left=0.1, right=0.66)
    fig = plt.figure(figsize=(55, 55))
    
    # Process and display images
    verbose_count = 0
    for j, title in enumerate(image_names):
        if verbose:
            ax = fig.add_subplot(gs[j//grid_size_y, j%grid_size_y])
        else:
            ax = fig.add_subplot(gs[verbose_count//grid_size_y, verbose_count%grid_size_y])
            
        if j == 0:
            # Raw image
            grayscaled_raw = tf.squeeze(image_list[j])*-1
            ax.imshow(grayscaled_raw, cmap="binary", alpha=1)
            ax.set_title(f"Step {j}: {title}")
            verbose_count += 1
        elif j >= len(image_names) - 2:
            # Contour visualization
            if j == len(image_names) - 1:
                ax.imshow(tf.squeeze(image_list[0]), alpha=0.75, cmap='gray')
            ax.imshow(tf.squeeze(image_list[j]), alpha=image_list[-2], cmap=CMAP1_STRICT)
            ax.set_title(f"Step {j}: {title}")
            verbose_count += 1
        elif verbose:
            ax.imshow(tf.squeeze(image_list[j]))
            ax.set_title(f"Step {j}: {title}")

    # Save and close
    plt.savefig(output_path)
    plt.close(fig)

def process_image_with_kernels(image: np.ndarray, kernels: list, threshold_value: float = 0.4,
                             alpha_bounds: tuple = (200, 750), inverse_alphas: bool = False,
                             output_dir: str = None, frame_num: int = None,
                             verbose: bool = False) -> tuple:
    """
    Process an image with a sequence of kernels and morphological operations

    Parameters
    ----------
    image : np.ndarray
        Input image array
    kernels : list
        List of kernels to apply
    threshold_value : float
        Threshold value for binarization
    alpha_bounds : tuple
        The row bounds between which the image will be checked for processing
    inverse_alphas : bool
        Whether to inverse the alpha values
    output_dir : str
        Directory to save output images
    frame_num : int
        Current frame number for saving
    verbose : bool
        Whether to show all processing steps

    Returns
    -------
    tuple
        (processed_image, contour_image, wave_positions)
    """
    # Normalize image
    wave_raw = image #/ 255.0 if image.max() > 1.0 else image
    # Convert the image to a PyTorch tensor
    if not isinstance(wave_raw, torch.Tensor):
        wave_raw = torch.tensor(wave_raw, dtype=torch.float32)

    # Store all processing steps
    image_list = [wave_raw]
    image_names = ["Raw Image"]

    # Add channel dimension if needed
    if len(wave_raw.shape) == 2:
        processed_image = tf.expand_dims(wave_raw, axis=0)  # Add batch dimension
        processed_image = tf.expand_dims(processed_image, axis=-1)  # Add channel dimension
    else:
        processed_image = tf.expand_dims(wave_raw, axis=0)  # Add batch dimension only
    # processed_image = tf.concat([tf.expand_dims(wave_raw, 3)], 3)

    # Add batch dimension
    # processed_image = tf.expand_dims(processed_image, axis=0)
    
    # Normalize values
    processed_image = tf.math.divide(
        tf.math.subtract(processed_image, tf.reduce_min(processed_image)),
        tf.math.subtract(tf.reduce_max(processed_image), tf.reduce_min(processed_image))
    )

    # Apply kernels
    for kernel in kernels:
        processed_image = filter_over_image(processed_image, kernel, normalize=True)#True)
        if verbose:
            image_list.append(processed_image)
            image_names.append(f"After {kernel if isinstance(kernel, str) else 'Gaussian'}")

    # Threshold
    processed_image = threshold_over_normalised_image(processed_image, threshold_value, 0)
    if verbose:
        image_list.append(processed_image)
        image_names.append("Thresholded")
    
    # Create alpha mask
    alphas = np.where(tf.squeeze(processed_image) > 0, 1, 0.0)
    
    # Handle alpha inversions and bounds
    if inverse_alphas:
        alphas = alphas.max() - alphas + alphas.min()
    alphas[:alpha_bounds[0], :] = 0
    alphas[alpha_bounds[1]:, :] = 0

    if verbose:
        image_list.append(alphas)
        image_names.append("Alpha Mask")

    # Apply morphological operations
    closing = cv2.morphologyEx(alphas, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    alphas = np.where(tf.squeeze(closing) > 0.0, 1, 0.0)

    if verbose:
        image_list.append(closing)
        image_names.append("After Morphological Operations")

    # Find wave contour
    contour = np.zeros(closing.shape)
    wave_positions = []
    prev_row = None

    # Track wave position
    for column in range(closing.shape[1]):
        val_changed = False
        for row in range(alpha_bounds[0], alpha_bounds[1]):
            if closing[row][column] > 0:
                contour[row][column] = 1
                val_changed = True
                prev_row = row
                break
        
        if prev_row is None:
            prev_row = alpha_bounds[0]
        if not val_changed:
            contour[prev_row][column] = 1
        wave_positions.append(prev_row)

    # Connect wave positions
    for pos in range(1, len(wave_positions)):
        height_change = wave_positions[pos] - wave_positions[pos-1]
        if height_change == 0:
            continue
        elif height_change < 0:
            contour[wave_positions[pos]:wave_positions[pos-1]+1, pos] = 1
        else:
            contour[wave_positions[pos-1]:wave_positions[pos]+1, pos] = 1

    # Add contour to visualization list
    image_list.append(contour)
    image_names.append("Wave Contour")
    # Add combined visualization
    image_list.append(contour)
    image_names.append("Combined")

    # Save visualization if output directory is provided
    if output_dir and frame_num is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'output_{frame_num:05d}.png')
        save_visualization(image_list, image_names, output_path, verbose)

    return processed_image, contour, wave_positions

def apply_kernels_batch(images: np.ndarray, used_kernels: list, threshold_value: float = 0.4,
                       alpha_bounds: tuple = (200, 750), inverse_alphas: bool = False,
                       output_dir: str = None, verbose: bool = False) -> tuple:
    """
    Process a batch of images with kernels and morphological operations

    Parameters
    ----------
    images : np.ndarray
        Array of images to process
    used_kernels : list
        List of kernels to apply
    threshold_value : float
        Threshold value for binarization
    alpha_bounds : tuple
        The row bounds between which the image will be checked for processing
    inverse_alphas : bool
        Whether to inverse the alpha values
    output_dir : str
        Directory to save output images
    verbose : bool
        Whether to show all processing steps

    Returns
    -------
    tuple
        (processed_images, contour_images, wave_positions)
    """
    processed_images = []
    contour_images = []
    all_wave_positions = []
    
    for i, image in enumerate(tqdm(images, desc="Processing images with kernels")):
        # if i>5000 and i<5002:
            # Preprocess image before applying kernels
            if isinstance(image, str):  # If image is a file path
                image = decode_image(image) / 255.0
            elif image.max() > 1.0:  # If image is not normalized
                image = image / 255.0

            # # Rotate image 90 degrees clockwise
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                
            processed, contour, positions = process_image_with_kernels(
                image,
                used_kernels,
                threshold_value,
                alpha_bounds,
                inverse_alphas,
                output_dir,
                i,
                verbose
            )
            processed_images.append(processed)
            contour_images.append(contour)
            all_wave_positions.append(positions)

    return np.array(processed_images), np.array(contour_images), all_wave_positions 