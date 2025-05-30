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

class torch_processor():

    def __init__(self):
        # Set matplotlib backend
        plt.rcParams["savefig.bbox"] = 'tight'

        # Colormap for visualization
        self.CMAP1_STRICT = plt.cm.colors.ListedColormap(['green'])

        # Dictionary of predefined kernels
        self.KERNELS = {
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

    def create_gaussian_filter(self, sigma: float = 1.4, k: int = 2) -> tf.Tensor:
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

    def create_sobel_kernel(self, k: int = 3, transposed: bool = False) -> tf.Tensor:
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

    def get_kernel(self, name: str = "Edge Detect") -> tf.Tensor:
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
        krnl = self.KERNELS[name]
        krnl = tf.reshape(krnl, [*krnl.shape, 1, 1])
        return tf.cast(krnl, dtype=tf.float32)

    def filter_over_image(self, input_image: np.ndarray, kernel: Union[tf.Tensor, str], normalize: bool = True) -> np.ndarray:
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
            krnl = self.get_kernel(kernel)
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

    def threshold_over_normalised_image(self, input_image: np.ndarray, threshold_percent: float, replacement_val: float) -> tf.Tensor:
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

    def save_visualization(self, image_list, image_titles, output_path, verbose=True):
        """
        Applies given kernels over an image, returns the output as an image and optinally saves it.

        Parameters
        ----------
        image_list : list of lists
            List of tensor objects representing the images of waves
            Example:
            image = [
                [raw_image1, processed_image1],
                [raw_image2, processed_image2],
                ...
                ]

        image_titles : list of strings
            The titles of the images. The total length of image_titles should be equal to the length of a set of images.
            Example:
            If an image list is = [raw_image1, processed_image1, processed_image2, processed_image3]
            Then the image_titles should be = ["Raw image", "Kernel 1", "Kernel 2", ""]

        output_path : path
            The path where the images are going to be saved to

        verbose : bool
            Should the in-between steps be shown/saved as well
        """

        if output_path:
            self.create_folders(output_path)
        for iteration in range(len(image_list)):
            # Set up the figure
            grid_size_x = math.floor((len(image_list[0]))**(1/2)) if verbose else 1
            grid_size_y = math.ceil(len(image_list[0])/grid_size_x) if verbose else 3
            
            gs = gridspec.GridSpec(grid_size_x, grid_size_y, bottom=0.1, top=0.45, left=0.1, right=0.66)
            fig = plt.figure(figsize=(55, 55))
            
            # Process and display images
            verbose_count = 0
            for j, title in enumerate(image_titles):
                if verbose:
                    ax = fig.add_subplot(gs[j//grid_size_y, j%grid_size_y])
                else:
                    ax = fig.add_subplot(gs[verbose_count//grid_size_y, verbose_count%grid_size_y])
                    
                if j == 0:
                    # Raw image
                    grayscaled_raw = tf.squeeze(image_list[iteration][j]*-1)
                    ax.imshow(grayscaled_raw, cmap="binary", alpha=1)
                    ax.set_title(f"Step {j}: {title}")
                    verbose_count += 1
                elif j >= len(image_titles) - 2:
                    # Contour visualization
                    if j == len(image_titles) - 1:
                        ax.imshow(tf.squeeze(image_list[iteration][0]), alpha=0.75, cmap='gray')
                    ax.imshow(tf.squeeze(image_list[iteration][-1]), alpha = image_list[iteration][len(image_titles)-2], cmap=self.CMAP1_STRICT)
                    ax.set_title(f"Step {j}: {title}")
                    verbose_count += 1
                elif verbose:
                    ax.imshow(tf.squeeze(image_list[iteration][j]))
                    ax.set_title(f"Step {j}: {title}")

            # Save and close
            # if save_image:
            plt.savefig(os.path.join(output_path, "frames") + f"/frame_{iteration:05d}.jpg")
            plt.close(fig)

    def process_image_with_kernels(self, used_kernels = ["Edge Detect"], images = "../data/incident_illumination", image_range = [271, 272], threshold_value=0.540, inverse_image_alphas=False, alpha_bounds=(100,553), morph_transforms=[]) -> tuple:
        """
        Applies given kernels over an image, returns the output as an image and optinally saves it.

        Parameters
        ----------
        used_kernels : list[str, tf.constant]
            List of kernels as either strings, tf.constants, or both.

        images : str | path | np.array
            This variable can either be:
                The path where the images are located.
                A numpy array of shape [number_of_images, height, width].

        image_range : list
            The range/list of numbers representing the current frame being processed
        
        threshold_value :float 
            A value, typically in the range [0,1], below which all values will be set to 0
        
        inverse_image_alphas :bool 
            Should the alpha (transparency) values be inversed. 
        
        save_image : bool
            Should the image be saved
        
        alpha_bounds : tuple
            The row bounds between which the image will be checked for processing.
            Useful since our images have lots of unused space.

        morph_transforms : list of tuple pairs
            A list of morphological transformations in the following format:
            list = [(cv2_morphological_transform, filter shape and values)]

            Example: 

            morph = [(cv2.MORPH_CLOSE, np.ones((3,3),np.uint8)]

        Returns
        -------

        full_image_list : list
            List of preprocessed images
            
        """
        full_image_list = []
        # Create a static list of non-linked objects
        for i in range(len(image_range)):
            full_image_list.append([tf.convert_to_tensor(np.ones((1024, 1024)), dtype=tf.float32)]*(len(used_kernels)+len(morph_transforms)+2))
        
        for i in range(len(image_range)):
            if i%500==0:
                print(f"Frame #{i}")
            # Used for image show
            if type(images) == np.ndarray:
                wave_raw = np.transpose(images[i]) # This should be transposed
                if len(wave_raw.shape) == 2:
                    wave_raw = tf.expand_dims(wave_raw, 0)
            else:
                wave_raw = decode_image(str(str(images) + f'/frame_{str(image_range[i]).zfill(5)}.jpg'))/255.0
            # image_list = [tf.squeeze(wave_raw)]
            full_image_list[i][0] = tf.squeeze(wave_raw)
            # Used for processing
            processed_image = tf.concat([tf.expand_dims(wave_raw, 3)], 3)
            processed_image = tf.math.divide(tf.math.subtract(
                                        processed_image, 
                                        tf.reduce_min(processed_image)
                                    ), 
                                    tf.math.subtract(
                                        tf.reduce_max(processed_image), 
                                        tf.reduce_min(processed_image)
                                    ))
            
            for num, krn in enumerate(used_kernels):
                processed_image = self.filter_over_image(processed_image, krn, normalize=True)
                full_image_list[i][num+1] = tf.squeeze(processed_image)

            # Thresholding
            processed_image = self.threshold_over_normalised_image(processed_image, threshold_value, 0)
            # image_list.append(processed_image)
            # image_names.append("thresholed")

            ###########################
            # Alpha edits for inverse #
            alphas = np.where(tf.squeeze(processed_image) > 0, 1, 0.0)
            if inverse_image_alphas:
                alphas = alphas.max() - alphas + alphas.min()
            alphas[:alpha_bounds[0], :] = 0
            # alphas[553:, :] = 0
            alphas[alpha_bounds[1]:, :] = 0
            ###########################
            ###########################
            
            #NOTE: Although this should be its own function as well, I do not wish to complicate things for future users. morphological transformations will be handled through a list, even if the order should be different
            ############ Morphological transformations ############
            closing = alphas
            for j in range(len(morph_transforms)):
                closing = cv2.morphologyEx(closing, morph_transforms[j][0], morph_transforms[j][1])
                full_image_list[i][j+len(used_kernels)+1] = tf.squeeze(closing)

            alphas = np.where(tf.squeeze(closing) > 0.0, 1, 0.0)
            #######################################################
            #######################################################
            full_image_list[i][-1] = tf.squeeze(alphas)

        return full_image_list

    def top_of_wave(self, images, adjecency = 10):
        """
        Finds the first detected edge within each column, returns a contour of the top of the waves

        Parameters
        ----------
        images : list of lists
            List of tensor objects representing the images of waves
            Example:
            image = [
                [raw_image1, processed_image1],
                [raw_image2, processed_image2],
                ...
                ]
        adjecency : int
            A value representing how close the next change in wave height must be.
            Example: int(0.01 * image[0].shape[1])
        Returns
        -------
        top_wave_images : list
            List of preprocessed images
        """
        # This is the second image full_image_list[image_range[0]-i][1] = tf.squeeze(alphas)
        top_wave_images = images
        right_most_coords_list = []
        for i in range(len(images)):
            closing = images[i][-1]
            # Rows, Columns indexes of conditional
            wheretops = np.where(closing > 0)
            # Creating top coordinates, with columns as keys
            top_coordinates = {}
            for index, col in enumerate(wheretops[1]):
                if top_coordinates.get(col) is None:
                    top_coordinates[int(col)] = int(wheretops[0][index])

            # Creating top "image"
            tops_image = np.zeros(closing.shape)
            prev_value = 0
            for col in range(closing.shape[1]):
                # Get from column dict, else prev value
                tops_image[top_coordinates.get(col, prev_value)][col] = 1

                # If height change then connect values
                if col != 0:
                    height_change = top_coordinates.get(col, prev_value) - prev_value
                    if height_change > 0:
                        tops_image[prev_value:top_coordinates.get(col)+1,col] = 1
                    elif height_change < 0:
                        tops_image[top_coordinates.get(col):prev_value+1,col] = 1
                    

                prev_value = top_coordinates.get(col, prev_value)

            # top_wave_images[i][1] = tops_image
            # Is this X, Y or Y, X? These are X, Y values i believe
            sorted_keys = sorted(top_coordinates.keys(), reverse=True)
            # adjecency = int(0.01 * tops_image.shape[1])
            adjecency_counter = 0
            right_most_coords_index = 0
            right_most_coords = (sorted_keys[0], top_coordinates.get(sorted_keys[0]))
            # Find right most coords whose following ADJECENCY_N steps are all within ADJECENCY val of each other
            for col_1 in range(1, len(sorted_keys)):
                if sorted_keys[col_1-1] - sorted_keys[col_1] <= adjecency:
                    adjecency_counter += 1
                else:
                    adjecency_counter = 0
                    right_most_coords_index = col_1
                    right_most_coords = (sorted_keys[right_most_coords_index], top_coordinates.get(sorted_keys[right_most_coords_index]))
                if adjecency_counter >= adjecency:
                    break

            # tops_image[:,right_most_coords[1]] = 1
            top_wave_images[i][-1] = tops_image
            right_most_coords_list.append(right_most_coords)
            # top_wave_images[i][1][:][max(top_coordinates.keys())] = np.ones(top_wave_images[i][1][:][max(top_coordinates.keys())].shape)


        return top_wave_images, right_most_coords_list


    def apply_avgpooling(self, image_range, images, kernel_size, stride, save_location, toe_positions = [(500, 800)]):
        """
        Apply average pooling filter over a collection of images, return as an image

        Parameters
        ----------
        image_range : list
            The collection of images
        images : str | path | np.array
            This variable can either be:
                The path where the images are located.
                A numpy array of shape [number_of_images, height, width].
        kernel_size : int 
            Size of the avgpooling layer kernel
        stride : int
            Step size for the average pooling
        save_location : boolean
            Path to the image save location
        toe_start : tuple(int, int)
            Current coordinates of the roller toe
        """
        toe_distance = toe_positions[0][0] # (X, Y) tuple
        image_sum = None
        # toe_images = []
        for i in image_range:
            if i%500==0:
                print(f"Frame #{i}")

            if type(images) == np.ndarray:
                wave_raw = images[i]
            else:
                wave_raw = decode_image(str(str(images) + f'/frame_{str(i).zfill(5)}.jpg'))
            # This needs to be changed to accept list of positions
            # current_toe_pos = int(math.floor(toe_distance + ((i-image_range[0])*(wave_raw.shape[2]-toe_distance)/(image_range[-1]-image_range[0]+1))))
            wave_raw = wave_raw.squeeze()
            current_toe_pos = toe_positions[i-image_range[0]][0] if toe_positions[i-image_range[0]][0] >= toe_distance else toe_distance
            wave_raw = wave_raw[:, max(current_toe_pos-toe_distance, 0):current_toe_pos] 

            if i==image_range[0]:
                image_sum = torch.zeros(wave_raw.shape[0], toe_distance)
            # This is annoying and literally just to transform Eagertensor from tensorflow into pytorch tensor
            wave_raw = torch.tensor(tf.cast(wave_raw, dtype=tf.float32).numpy())
            image_sum = image_sum.add(wave_raw)
            # wave_raw2 = decode_image(str(image_loc / f'{str(i).zfill(5)}.jpg'))/255.0

        image_list = [image_sum/(image_range[-1]-image_range[0]+1)]
        image_names = [f"Mean Image of {image_range[-1]-image_range[0]} frames"] # AVG pooling still has to be added
        # Used for processing
        processed_image = image_list[0]
        avgpool = nn.AvgPool2d(kernel_size, stride)
        processed_image = avgpool(processed_image.unsqueeze(0))
        image_list.append(processed_image)
        image_names.append(f"Avg pooling: kernel size {kernel_size}, stride {stride}")


        grid_size_x = math.floor((len(image_list))**(1/2))
        grid_size_y = math.ceil((len(image_list))/grid_size_x)  
        gs = gridspec.GridSpec(grid_size_x, grid_size_y, bottom=0.1, top=0.45, left=0.1, right=0.66)   
        fig = plt.figure()
        fig.set_figwidth(55)
        fig.set_figheight(55)
        
        # Process image
        for j, krn in enumerate(image_names):
            # Display image
            ax = fig.add_subplot(gs[(j)//grid_size_y, (j)%grid_size_y])
            ax.imshow(tf.squeeze(image_list[j])*-1, cmap="binary")
            if j == len(image_names)-1:
                for k, l in np.ndenumerate(image_list[j].squeeze()):
                    ax.text(k[1], k[0], '{:0.1f}'.format(l), ha='center', va='center', fontsize=8, color=f"{int(l<=127)}")
            ax.set_title(f"Step {j}: {krn}")
        
        plt.savefig(save_location + f"/output_{i}.png") # Multi kernels
        plt.close()

    def read_npz_as_numpy(self, path):
        return np.load(path)['arr_0']
    
    def _combine_gray_and_rgb(self, gray, rgb, color=1):
        expanded = np.expand_dims(gray, axis=2) # Red
        expanded = np.append(expanded, np.expand_dims(gray, axis=2), axis=2) # Green
        expanded = np.append(expanded, np.expand_dims(gray, axis=2), axis=2) # Blue

        # RGB 
        for i in range(3):
            expanded[:, :, i] = np.where(rgb == 1, int(i==color), expanded[:, :, i])
        
        return expanded
    
    def npz_to_video(self, save_path, npz_path, preprocessed_path, start_frame, end_frame, roller_toe=None):
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (1024, 1024), True)
        for i in range(start_frame, end_frame):
            # Combine 
            color = 1 # Color = rgb. Red = 0, Green = 1, Blue = 2
            npz_im = self.read_npz_as_numpy(npz_path + f"/frame_{(i):05d}.npz")
            preproc_im = tf.squeeze(decode_image(str(str(preprocessed_path) + f'/frame_{(i):05d}.jpg'))/255.0)
            combined = self._combine_gray_and_rgb(preproc_im, npz_im, color) 
            if roller_toe:
                combined[:, roller_toe[i-start_frame][0], color] = 1
            combined = (combined*255).astype(np.uint8)
            out.write(combined)

        out.release()

    def create_folders(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "frames"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "combined"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "pytorch_dir_top_of_wave"), exist_ok=True)


