"""
Image processing module for MRAW video analysis.
Provides both traditional OpenCV-based and PyTorch-based processing capabilities.
"""

import cv2
import numpy as np
from skimage import img_as_ubyte
from tqdm import tqdm
import os
from torch_processor import process_image_with_kernels
import torch
import tensorflow as tf
from torchvision.io import decode_image

def normalize_and_clahe(img, clip_limit=10.0, tile_grid_size=(8,8)):
    """
    Normalize image and apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        img (numpy.ndarray): Input image
        clip_limit (float): CLAHE clip limit
        tile_grid_size (tuple): CLAHE tile grid size
        
    Returns:
        numpy.ndarray: Processed image
    """
    img_norm = img_as_ubyte(img / img.max())
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_norm)

def process_incident_illumination(images, output_dir=None, background_frame_idx=0, process_frame_idx=None, 
                                use_torch=False, torch_kernels=None):
    """
    Process incident illumination video frames
    
    Args:
        images (numpy.ndarray): Image array
        output_dir (str, optional): Directory to save processed images
        background_frame_idx (int): Index of frame to use as background
        process_frame_idx (int, optional): Index of frame to process, if None, processes all frames
        use_torch (bool): Whether to use PyTorch-based processing
        torch_kernels (list): List of kernel names to apply if use_torch is True
        
    Returns:
        tuple: List of processed images with their titles
    """
    # Create output directories if specified
    if output_dir:
        frame_dir = os.path.join(output_dir, "frames")
        combined_dir = os.path.join(output_dir, "combined")
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(combined_dir, exist_ok=True)
    
    # Process background frame
    background = cv2.rotate(
        normalize_and_clahe(images[background_frame_idx]), 
        cv2.ROTATE_90_CLOCKWISE
    )
    
    # Determine frames to process
    if process_frame_idx is not None:
        frames_to_process = [process_frame_idx]
    else:
        frames_to_process = range(len(images))
    
    # Process each frame
    images_list = None
    for i in tqdm(frames_to_process, desc="Processing incident illumination frames"):
        # Process current frame
        frame = cv2.rotate(
            normalize_and_clahe(images[i]), 
            cv2.ROTATE_90_CLOCKWISE
        )
        
        # Calculate difference
        diff = cv2.absdiff(background, frame)
        
        if use_torch and torch_kernels:
            # Apply PyTorch-based processing
            processed, contour, positions = process_image_with_kernels(
                frame,#diff,
                torch_kernels,
                threshold_value=0.4,
                alpha_bounds=(200, 750),
                output_dir=output_dir if output_dir else None,
                frame_num=i,
                verbose=False
            )

            # Convert tensor processed back to array
            if isinstance(processed, torch.Tensor):
                processed = processed.detach().cpu().numpy()
            elif isinstance(processed, tf.Tensor):
                processed = processed.numpy()
            
            result = processed
        else:
            # Apply traditional processing
            clahe_img_noise = normalize_and_clahe(diff) #For first frame=background we will get warning since all 0s
            diff_blur = cv2.GaussianBlur(diff, (3, 3), 0)
            _, thresholded = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)
            result = cv2.bitwise_and(diff, diff, mask=thresholded)
        
        # Apply CLAHE to final result
        clahe_img = normalize_and_clahe(result)
        
        # Save individual frame
        if output_dir:
            frame_path = os.path.join(frame_dir, f"frame_{str(i).zfill(5)}.jpg")
            cv2.imwrite(frame_path, clahe_img)
        
        # Save combined image for visualization (middle frame)
        if i == len(frames_to_process) // 2 and output_dir:
            # Prepare list of processed images
            images_list = [
                ("Background", background),
                ("Wave", frame),
                ("Difference", diff),
                ("Processed", result),
                ("CLAHE Final", clahe_img),
            ]
            combined_path = os.path.join(combined_dir, f'combined_image_incident_illumination_{i}.png')
            create_combined_image(images_list, display_image, save_name=combined_path)
    
    return images_list

def process_back_illumination(images, output_dir=None, background_frame_idx=0, start_idx=None, end_idx=None,
                            use_torch=False, torch_kernels=None, vis_dir=None, verbose=False):
    """
    Process back illumination video frames using the notebook's approach
    
    Args:
        images (numpy.ndarray): Image array
        output_dir (str, optional): Directory to save processed images
        background_frame_idx (int): Index of frame to use as background
        start_idx (int, optional): Start frame index, if None, use 0
        end_idx (int, optional): End frame index, if None, process all frames
        use_torch (bool): Whether to use PyTorch-based processing
        torch_kernels (list): List of kernel names to apply if use_torch is True
        vis_dir (str, optional): Directory to save visualizations when using torch processing
        verbose (bool): Whether to show all processing steps in visualizations
        
    Returns:
        tuple: Last frame's processed images with their titles
    """
    # Define parameters
    gamma = 2.5
    contrast_dark = 0.15  # Increased contrast for dark areas
    contrast_bright = 0.3  # Lowered contrast for bright areas
    threshold = 128
    gain = 1.8
    
    # Create output directories if specified
    if output_dir:
        frame_dir = os.path.join(output_dir, "frames")
        combined_dir = os.path.join(output_dir, "combined")
        contour_dir = os.path.join(output_dir, "contours")  # New directory for contours
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(combined_dir, exist_ok=True)
        os.makedirs(contour_dir, exist_ok=True)
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
    
    # Set frame range
    start_idx = 0 if start_idx is None else start_idx
    end_idx = len(images) if end_idx is None else end_idx
    
    # Precompute gamma correction lookup table
    gamma_correction_table = np.array([(i / 255.0) ** (1 / gamma) * 255 for i in np.arange(256)]).astype("uint8")
    
    # Process background frame (first frame)
    img_rotated_background = None
    
    # Calculate middle frame for combined visualization
    middle_frame_idx = (start_idx + end_idx) // 2
    
    # Define helper functions for processing
    def adaptive_contrast(img, threshold):
        mask_bright = img > threshold
        mask_dark = ~mask_bright
        img_adjusted = np.zeros_like(img, dtype=np.float32)
        img_adjusted[mask_bright] = img[mask_bright] * contrast_bright
        img_adjusted[mask_dark] = img[mask_dark] * contrast_dark
        return np.clip(img_adjusted, 0, 255).astype(np.uint8)
    
    def process_frame(img):
        img_32bit = img.astype('float32')
        img_gamma_corrected = cv2.LUT(cv2.convertScaleAbs(img_32bit), gamma_correction_table)
        img_contrast_adaptive = adaptive_contrast(img_gamma_corrected, threshold)
        img_hdr = np.clip(img_contrast_adaptive * gain, 0, 255).astype(np.uint8)
        return cv2.rotate(img_hdr, cv2.ROTATE_90_CLOCKWISE)
    
    # Process frames one by one, as in the notebook
    last_result = None
    wave_positions = []  # Store wave positions for each frame
    
    for i in tqdm(range(start_idx, end_idx), desc="Processing back illumination frames"):
        # Process background frame if this is the first frame
        if i == start_idx or img_rotated_background is None:
            img_rotated_background = process_frame(images[background_frame_idx])
        
        # Process current frame
        img_rotated = process_frame(images[i])
        
        # Calculate difference
        diff = cv2.absdiff(img_rotated_background, img_rotated)
        
        if use_torch and torch_kernels:

            # Preprocess image before applying kernels
            if isinstance(images[i], str):  # If image is a file path
                image = decode_image(images[i]) / 255.0
            elif images[i].max() > 1.0:  # If image is not normalized
                # image = images[i] / 255.0
                # Normalize to 0-1 range
                max_val = images[i].max()
                image = images[i] / max_val
                # print("Max after",np.max(image))

            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Apply PyTorch-based processing with all steps from notebook
            processed, contour, positions = process_image_with_kernels(
                image,#s[i],
                torch_kernels,
                threshold_value=0.54,
                alpha_bounds=(300,650),#(100, 553),
                output_dir=vis_dir if vis_dir else None,
                frame_num=i,
                verbose=verbose
            )
            
            # Convert tensor processed back to array
            if isinstance(processed, torch.Tensor):
                processed = processed.detach().cpu().numpy()
            elif isinstance(processed, tf.Tensor):
                processed = processed.numpy()
            
            # # Ensure the array is properly normalized
            # if processed.max() <= 1.0:
            #     processed = (processed * 255).astype(np.uint8)
            result = processed
            wave_positions.append(positions)
            
            # Save contour visualization
            if output_dir:
                contour_path = os.path.join(contour_dir, f"contour_{str(i).zfill(5)}.jpg")
                cv2.imwrite(contour_path, (contour * 255).astype(np.uint8))
                
                # Save processed frame
                frame_path = os.path.join(frame_dir, f"frame_{str(i).zfill(5)}.jpg")
                cv2.imwrite(frame_path, (processed * 255).astype(np.uint8))
        else:
            # Apply traditional processing as in the notebook
            # Apply CLAHE to difference image (with noise)
            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
            clahe_img_noise = clahe.apply(diff)
            
            # Remove noise
            diff_blur = cv2.GaussianBlur(diff, (3, 3), 0)
            _, thresholded = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)
            result = cv2.bitwise_and(diff, diff, mask=thresholded)
            
            # Apply CLAHE to final result
            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(result)
            
            # Save frame
            if output_dir:
                frame_path = os.path.join(frame_dir, f"frame_{str(i).zfill(5)}.jpg")
                cv2.imwrite(frame_path, result)
        
        # Create visualization for middle frame
        if i == middle_frame_idx and output_dir:
            # Create list of all processing steps for visualization
            images_list = [
                ("Background", img_rotated_background),
                ("Wave", img_rotated),
                ("Difference", diff),
            ]
            
            if use_torch and torch_kernels:
                images_list.extend([
                    ("Processed", result),
                    ("Wave Contour", contour * 255)
                ])
            else:
                images_list.extend([
                    ("CLAHE Noise Check", clahe_img_noise),
                    ("Noise Removed", result),
                    ("CLAHE Final", clahe_img)
                ])
            
            # Save combined image
            combined_path = os.path.join(combined_dir, f'combined_image_back_illumination_frame_{str(i).zfill(5)}.png')
            create_combined_image(images_list, display_image, save_name=combined_path)
            
            # Store visualization from middle frame
            last_result = images_list
        
        # If this is the last frame and we haven't stored a visualization yet
        if i == end_idx - 1 and last_result is None:
            if use_torch and torch_kernels:
                last_result = [
                    ("Background", img_rotated_background),
                    ("Wave", img_rotated),
                    ("Difference", diff),
                    ("Processed", result),
                    ("Wave Contour", contour * 255)
                ]
            else:
                last_result = [
                    ("Background", img_rotated_background),
                    ("Wave", img_rotated),
                    ("Difference", diff),
                    ("CLAHE Noise Check", clahe_img_noise),
                    ("Noise Removed", result),
                    ("CLAHE Final", clahe_img)
                ]
    
    # Save wave positions if using torch processing
    if output_dir and use_torch and wave_positions:
        np.save(os.path.join(output_dir, 'wave_positions.npy'), np.array(wave_positions))
    
    return last_result

def display_image(title, img, save_path=None):
    """
    Display image with OpenCV and optionally save it
    
    Args:
        title (str): Window title
        img (numpy.ndarray): Image to display
        save_path (str, optional): Path to save the image
    """
    # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    # Uncomment the following lines to show the image
    # cv2.imshow(title, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    
    if save_path:
        cv2.imwrite(save_path, img)

def create_combined_image(images_list, display_func, save_name=None):
    """
    Create a combined image with multiple processed images and titles
    
    Args:
        images_list (list): List of (title, image) tuples
        display_func (function): Function to display/save the image
        save_name (str, optional): Path to save the combined image
        
    Returns:
        numpy.ndarray: Combined image
    """
    # Ensure all images are 3-channel (convert grayscale to BGR)
    images_list = [(title, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img) 
                   for title, img in images_list]

    # Ensure all images have the same height
    height = min(img.shape[0] for _, img in images_list)
    resized_images = [(title, cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))) 
                      for title, img in images_list]

    # Create a blank space for titles
    title_height = 100
    title_image = np.ones((title_height, sum(img.shape[1] for _, img in resized_images), 3), dtype=np.uint8) * 255

    # Add titles to the blank space
    x_offset = 0
    for title, img in resized_images:
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_x = x_offset + (img.shape[1] - text_size[0]) // 2
        text_y = (title_height + text_size[1]) // 2
        cv2.putText(title_image, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        x_offset += img.shape[1]

    # Concatenate the title image with the actual images
    combined_image = np.vstack([title_image, np.hstack([img for _, img in resized_images])])

    # Save and display
    if save_name:
        display_func('combined_images', combined_image, save_name)
    
    return combined_image 