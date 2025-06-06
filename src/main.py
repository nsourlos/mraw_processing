#!/usr/bin/env python3
"""
Example script for using the MRAW Processor toolkit

This script demonstrates how to use the key functions from the toolkit
in a simple workflow, including both traditional and PyTorch-based processing.
It determines illumination type based on folder names.
"""

import os
import sys
import glob
import numpy as np
import time
import logging
import cv2
from datetime import datetime
from termcolor import colored
# from mraw_loader import load_mraw_video, save_to_hdf5, load_from_hdf5, load_from_npy
from mraw_loader import mraw_loader
# from image_processor import process_incident_illumination, process_back_illumination
from image_processor import image_processor
# from column_analyzer import analyze_columns, create_animation, get_wave_roller
from column_analyzer import column_analyzer
# from torch_processor import create_gaussian_filter,  apply_kernels_batch
from torch_processor import torch_processor

# Configure logging
def setup_logging(output_dir):
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mraw_processor_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def timer(func):
    def _wrapper(*args, **kwargs):
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return _wrapper

def determine_illumination_type(folder_path):
    """
    Determine illumination type based on folder name
    
    Args:
        folder_path (str): Path to folder
        
    Returns:
        str: 'back' or 'incident' illumination type
    """
    folder_name = os.path.basename(folder_path).lower()
    if 'back' in folder_name:
        return 'back'
    elif 'incident' in folder_name:
        return 'incident'
    else:
        # Default to incident if can't determine
        print(f"Warning: Could not determine illumination type from folder name '{folder_name}'. Defaulting to 'incident'.")
        logging.warning(f"Could not determine illumination type from folder name '{folder_name}'. Defaulting to 'incident'.")
        return 'incident'

# Main func should be kept to a minimum.
def main():
    # Start total script timing
    total_start_time = time.time()
    
    # Define paths
    # This goes one layer above for me.
    input_dir = "data/"  # Root directory for data (one folder behind)
    output_dir = "output/"  # Where to save processed results (one folder behind)

    # Define class objects
    col_analyzer = column_analyzer()
    im_proc = image_processor()
    mraw = mraw_loader()
    t_processor = torch_processor()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(output_dir)
    print(f"Log file created at: {log_file}")
    logging.info(f"Log file created at: {log_file}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        logging.error(f"Input directory {input_dir} does not exist.")
        sys.exit(1)
    
    # Find all subfolders in the input directory
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    
    if not subfolders:
        print(f"No subfolders found in {input_dir}")
        logging.error(f"No subfolders found in {input_dir}")
        sys.exit(1)
    
    # Keep track of h5 files created during processing
    created_h5_files = set()
    
    for folder in subfolders:
        # Start folder timing
        folder_start_time = time.time()
        
        # Determine illumination type from folder name # do we not want this to be determined by the user?
        illumination_type = determine_illumination_type(folder)
        print(f"\n{colored('Step 1: Processing folder: ' + folder + f' (Detected illumination type: {illumination_type})', 'green')}")
        logging.info(f"Step 1: Processing folder: {folder} (Detected illumination type: {illumination_type})")
        
        # Find all files to process in this folder
        cihx_files = glob.glob(os.path.join(folder, "**", "*.cihx"), recursive=True)
        h5_files = glob.glob(os.path.join(folder, "**", "*.h5"), recursive=True)
        npy_files = glob.glob(os.path.join(folder, "**", "*.npy"), recursive=True)
        
        # Filter out h5 files that were created during processing
        h5_files = [f for f in h5_files if f not in created_h5_files] # why? In case of multiple h5 files?
        
        all_files = cihx_files + h5_files + npy_files
        
        if not all_files:
            print(f"No files found in {folder}")
            logging.warning(f"No files found in {folder}")
            continue
            
        print(f"Found {len(all_files)} files to process")
        logging.info(f"Found {len(all_files)} files to process")
        print("\n")
        logging.info("")
        
        for input_file in all_files:
            # Start file timing
            file_start_time = time.time()
            
            folder_name = os.path.basename(folder)
            print('Folder name:', folder_name)
            logging.info(f'Folder name: {folder_name}')
            file_name = os.path.basename(input_file).split('.')[0]
            print("File name:", file_name)
            logging.info(f"File name: {file_name}")
            
            # Step 2: Convert MRAW to HDF5 if needed # This seems to be always done. A check is in order.
            if False:
                if input_file.endswith('.cihx'):
                    if (input_file[:-5] + ".h5") in all_files:
                        print(f"h5 file: {input_file[:-5]} already exists")
                        logging.info(f"h5 file: {input_file[:-5]} already exists")
                        continue
                    print(colored(f"Step 2: Converting {input_file} to HDF5", 'green'))
                    logging.info(f"Step 2: Converting {input_file} to HDF5")
                    # hdf5_file = os.path.join(output_dir, f"{folder_name}_{file_name}.h5")

                    # Save h5 file in the same location as the input file
                    hdf5_file = os.path.join(os.path.dirname(input_file), f"{file_name}.h5")
                    images, _ = mraw.load_mraw_video(input_file)
                    if images is not None:
                        mraw.save_to_hdf5(images, hdf5_file, compression_level=1)
                        # Add the newly created h5 file to our tracking set
                        created_h5_files.add(hdf5_file)
                        print("Files to ignore:",created_h5_files)
                        logging.info(f"Files to ignore: {created_h5_files}")
                        # Use the HDF5 file for further processing
                        input_file = hdf5_file
                    else:
                        print(f"Error loading MRAW file: {input_file}")
                        logging.error(f"Error loading MRAW file: {input_file}")
                        continue
                
                # Step 2: Load the file for processing #NOTE: if we have an h5 and an npy file, both will be loaded which takes time
                elif input_file.endswith('.h5') and input_file not in created_h5_files:
                    print(colored(f"Step 2: Loading from file {input_file}", 'green'))
                    logging.info(f"Step 2: Loading from file {input_file}")
                    images = mraw.load_from_hdf5(input_file)
                elif input_file.endswith('.npy'):
                    print(colored(f"Step 2: Loading from file {input_file}", 'green'))
                    logging.info(f"Step 2: Loading from file {input_file}")
                    images = mraw.load_from_npy(input_file)
                else:
                    if input_file in created_h5_files:
                        print(f"Skipping file {input_file} (already processed)")
                        logging.info(f"Skipping file {input_file} (already processed)")
                    else:
                        print(f"Unsupported file type: {input_file}")
                        logging.warning(f"Unsupported file type: {input_file}")
                    continue
                    
                if images is None:
                    print(f"Error loading images from {input_file}")
                    logging.error(f"Error loading images from {input_file}")
                    continue
                    
                print("\n")
                logging.info("")
            
            # Step 3: Process images based on illumination type
            process_dir = os.path.join(output_dir, f"{folder_name}_{file_name}")
            traditional_dir = os.path.join(process_dir, "traditional")
            pytorch_dir = os.path.join(process_dir, "pytorch")
            os.makedirs(traditional_dir, exist_ok=True)
            os.makedirs(pytorch_dir, exist_ok=True)
            first_frame = 5676
            last_frame = 7308
            # images = None
            
            if illumination_type == 'back':
                # Process with back illumination methods
                if False:
                    print(colored("Step 3: Processing with back illumination...", 'green'))
                    logging.info("Step 3: Processing with back illumination...")
                    use_torch=False
                    
                    # # Traditional processing
                    print("Processing with traditional methods...")
                    logging.info("Processing with traditional methods...")
                    im_proc.process_back_illumination(
                        images,
                        output_dir=traditional_dir,
                        background_frame_idx=0,
                        start_idx=0,
                        end_idx=len(images),
                        use_torch=False
                    )
                
                
            else:  # incident illumination
                # Process with incident illumination methods
                if False:
                    print(colored("Processing with incident illumination...", 'green'))
                    logging.info("Processing with incident illumination...")
                    
                    # Traditional processing
                    # use_torch=False
                    print("Processing with traditional methods...")
                    logging.info("Processing with traditional methods...")
                    im_proc.process_incident_illumination(
                        images,
                        output_dir=traditional_dir,
                        background_frame_idx=0,
                    )
                
            
            # Free memory
            # del images
            print("\n")
            logging.info("")

            # Processing wave shape
            print("\n")
            logging.info("")
            print(colored("Creating and storing image contours...", 'green'), illumination_type)
            logging.info(f"Creating and storing image contours... {illumination_type}")
            # This only creates two images for some reason
            images_processed = t_processor.process_image_with_kernels(
                used_kernels=[t_processor.create_gaussian_filter(1.4,2), "Edge Detect"],
                images=str(os.path.join(traditional_dir, "frames")), # needs to work with set of images
                # images="output/incident_illumination_Side_View_Video_128mm_MRAW_015_C001H001S0001/traditional/combined/",
                image_range = range(first_frame, last_frame),
                threshold_value=0.50,
                inverse_image_alphas=False,
                alpha_bounds=(200, 750),
                morph_transforms=[(cv2.MORPH_CLOSE, np.ones((3,3),np.uint8)), (cv2.MORPH_OPEN, np.ones((3,3),np.uint8))]
                # morph_transforms=[]
            )

            t_processor.save_visualization(
                image_list=images_processed,
                # image_titles=["Raw Image", "Gaussian filter", "Edge detect", "Combined"],
                image_titles=["Raw Image", "Gaussian filter", "Edge detect", "Morph Close", "Morph Open", "Combined"],
                output_path=pytorch_dir,
                verbose=True
            )
            print("Images Processed. Finding top-of-wave...\n")
            t1 = time.time()
            images_processed, right_most_coordinates = t_processor.top_of_wave(images=images_processed, adjecency=int(images_processed[0].shape[1]))
            print(f"Found top of wave in {time.time() - t1} seconds. Saving images as npz...")
            pytorch_dir_top_of_wave = os.path.join(pytorch_dir, "top_of_wave") # I want this changed from contours to matrix or 2d tensor
            if pytorch_dir_top_of_wave:
                os.makedirs(pytorch_dir_top_of_wave, exist_ok=True)
            t1 = time.time()
            for frame_num, im_proc in enumerate(images_processed):
                np.savez_compressed(pytorch_dir_top_of_wave + f"/frame_{(frame_num+first_frame):05d}.npz", im_proc[1]) # Ideally this should work
            print(f"Saving complete. Saving files took {time.time() - t1} seconds.")
            # t_processor.save_visualization(
            #     image_list=images_processed,
            #     image_titles=["Raw Image", "Modified contour image", "Combined"],
            #     output_path=pytorch_dir,
            #     verbose=False
            # )
            t_processor.npz_to_video("output/test_roller_tracking.mp4",pytorch_dir_top_of_wave, os.path.join(traditional_dir,"frames"), first_frame, last_frame, right_most_coordinates)
            print("Starting frame:", first_frame, "End frame:", last_frame)
            logging.info(f"Starting frame: {first_frame}, End frame: {last_frame}")
            # del images_processed
            
            # Step 4: Analyze wave heights
            # Use frames from PyTorch processing when torch is enabled
            # frames_dir = os.path.join(pytorch_dir if use_torch else traditional_dir, "frames")
            # frames_dir_list = [os.path.join(pytorch_dir, "top_of_wave"), os.path.join(traditional_dir,"frames")]
            frames_dir_list = [os.path.join(traditional_dir,"frames"), pytorch_dir_top_of_wave]
            # frames_dir_list = [pytorch_dir_top_of_wave]
            analysis_dir = os.path.join(process_dir, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            output_path = os.path.join(traditional_dir,"combined")

            if False:
                if os.path.exists(os.path.join(traditional_dir, "frames")):
                    print(colored("Step 4: Analyzing wave heights...", 'green'))
                    logging.info("Step 4: Analyzing wave heights...")
                    bin_arrays, column_positions, frame_count= col_analyzer.analyze_columns(
                        os.path.join(traditional_dir,"frames"), 
                        num_columns=5, 
                        output_dir=output_path, 
                        max_frames=None,
                        skip_top_pixels=190,
                        middle_wave_height=500,
                        threshold_level=10 if illumination_type == 'back' else 30,
                        column_width=1,
                        bin_start=100, 
                        bin_end=200
                    )

                    print(colored("Creating animation...", 'green'))
                    logging.info("Creating animation...")
                    col_analyzer.create_animation(
                        os.path.join(traditional_dir,"frames"), 
                        output_path, 
                        bin_arrays, 
                        column_positions, 
                        frame_count,
                        num_columns=5, 
                        fps=60, #was 10 but resulting video was 34 mins, 300 strange artifacts
                        max_frames=None,
                        column_width=1,
                    )

                    del bin_arrays
            # backup_starting_frames = 0
            # backup_ending_frames = 25000
            for it, frames_dir in enumerate(frames_dir_list):
                output_path = os.path.join(analysis_dir, str(it))
                os.makedirs(output_path, exist_ok=True)
                if os.path.exists(frames_dir):
                    # Wave tracking implementation is incompatible with my implementation. Either I change a lot of it, or create a different Roller_toe func
                    # Optional: Wave roller tracking
                    print("\n")
                    logging.info("")
                    print(colored("Step 5: Obtaining wave roller...", 'green'), illumination_type)
                    logging.info(f"Step 5: Obtaining wave roller... {illumination_type}")
                    # This will be seperated into two: Nikos' contours and Josip's contours
                    wave_roller_coords, starting_frame, end_frame = col_analyzer.get_wave_roller(
                        frames_dir, 
                        output_path, 
                        starting_frame=0,
                        save_arrays=False,  # Save frames and contours as numpy arrays,
                        threshold_area=100000 if illumination_type == 'back' else 50000,
                        threshold_img=10 if illumination_type == 'back' else 100,
                        illumination_type=illumination_type
                    )
                    print("Starting frame:", starting_frame, "End frame:", end_frame)
                    logging.info(f"Starting frame: {starting_frame}, End frame: {end_frame}")

                    # Optional: Wave mean   
                    print("\n")
                    logging.info("")
                    print(colored("Step 6: Creating wave mean...", 'green'), illumination_type)
                    logging.info(f"Step 6: Creating wave mean... {illumination_type}")
                    # 
                    coordinates = wave_roller_coords if "traditional" in frames_dir else right_most_coordinates # interesting that both techniques create 1632 frames
                    starting_frame = starting_frame if "traditional" in frames_dir else  first_frame # should the XY coords be fixed?
                    end_frame = end_frame if "traditional" in frames_dir else last_frame
                    if len(coordinates) > 0:
                        t_processor.apply_avgpooling(image_range=range(starting_frame, end_frame), images=os.path.join(traditional_dir,"frames"), kernel_size=20, stride=20, save_location=output_path, toe_positions=coordinates)
                        print("Starting frame:", starting_frame, "End frame:", end_frame)
                        logging.info(f"Starting frame: {starting_frame}, End frame: {end_frame}")
                    else:
                        print("\n")
                        logging.info("")
                        print(colored("Wave roller not found.", 'green'), illumination_type)
                        logging.error(f"Wave roller not found. {illumination_type}")

                    first_frame = starting_frame
                    last_frame = end_frame


                else:
                    print(f"Warning: Frames directory not found: {frames_dir}")
                    logging.warning(f"Warning: Frames directory not found: {frames_dir}")
        
        
            # Calculate and print file processing time
            file_end_time = time.time()
            file_duration = file_end_time - file_start_time
            print(f"\n{colored(f'File processing time: {file_duration:.2f} seconds', 'red')}")
            logging.info(f"\nFile processing time: {file_duration:.2f} seconds")

            # Delete frames_dir and all debug_frame_*.jpg images inside analysis_dir
            if os.path.exists(frames_dir):
                print(f"\nDeleting frames directory: {frames_dir}")
                logging.info(f"\nDeleting frames directory: {frames_dir}")
                for root, dirs, files in os.walk(frames_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(frames_dir)
                
            if os.path.exists(analysis_dir):
                print(f"Deleting debug_frame_*.jpg images in analysis directory: {analysis_dir}")
                logging.info(f"Deleting debug_frame_*.jpg images in analysis directory: {analysis_dir}")
                for file_name in os.listdir(analysis_dir):
                    if file_name.startswith("debug_frame_") and file_name.endswith(".jpg"):
                        os.remove(os.path.join(analysis_dir, file_name))

        # Calculate and print folder processing time
        folder_end_time = time.time()
        folder_duration = folder_end_time - folder_start_time
        print(f"\n{colored(f'Folder processing time: {folder_duration:.2f} seconds', 'red')}")
        logging.info(f"\nFolder processing time: {folder_duration:.2f} seconds")

    # Calculate and print total script duration
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n{colored('Processing complete!', 'green')}")
    print(f"{colored(f'Total script duration: {total_duration:.2f} seconds', 'red')}")
    print(f"Results are saved in: {output_dir} \n")
    logging.info("\nProcessing complete!")
    logging.info(f"Total script duration: {total_duration:.2f} seconds")
    logging.info(f"Results are saved in: {output_dir} \n")

    # Processing info

if __name__ == "__main__":
    main()