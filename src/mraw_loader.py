import pyMRAW
import h5py
import os
import gc
import time
import numpy as np

class mraw_loader():
    def load_mraw_video(self, file_path):
        """
        Load MRAW video file using pyMRAW
        
        Args:
            file_path (str): Path to the MRAW (cihx) file
            
        Returns:
            tuple: (images array, metadata dictionary)
        """
        print(f"Loading: {file_path}")
        try:
            images, info = pyMRAW.load_video(file_path)
            print(f"Loaded {len(images)} frames with shape {images[0].shape}")
            return images, info
        except Exception as e:
            print(f"Error loading video: {e}")
            return None, None

    def save_to_hdf5(self, images, filename, compression_level=1):
        """
        Save images array to HDF5 file with compression
        
        Args:
            images (numpy.ndarray): Image array to save
            filename (str): Path to save the HDF5 file
            compression_level (int): GZIP compression level (1-9)
                Lower values (1-3) → Faster saving, larger files
                Medium values (4-6) → Balanced speed & size
                Higher values (7-9) → Smallest file size, slowest save time
        
        Returns:
            tuple: (file_size_mb, save_time_seconds)
        """
        start = time.time()
        
        try:
            print("Saving to HDF5...")
            with h5py.File(filename, "w") as f:
                f.create_dataset("array", data=images, compression="gzip", compression_opts=compression_level)
            
            save_time = time.time() - start
            file_size = os.path.getsize(filename) / (1024**2)  # Convert to MB
            
            print(f"Saved: {filename} | {file_size:.2f} MB | {save_time:.2f}s")
            return file_size, save_time
        
        except Exception as e:
            print(f"Error saving to HDF5: {e}")
            return 0, 0

    def load_from_hdf5(self, filename):
        """
        Load images array from HDF5 file
        
        Args:
            filename (str): Path to the HDF5 file
            
        Returns:
            numpy.ndarray: Loaded image array
        """
        print(f"Loading from HDF5: {filename}")
        start = time.time()
        
        try:
            with h5py.File(filename, "r") as f:
                data = f["array"][:]
            
            load_time = time.time() - start
            print(f"Loaded {len(data)} frames in {load_time:.2f}s")
            return data
        
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None
        
    def load_from_npy(self, filename):
        """
        Load images array from NPY file
        
        Args:
            filename (str): Path to the NPY file
            
        Returns:
            numpy.ndarray: Loaded image array
        """
        print(f"Loading from NPY: {filename}")
        start = time.time()
        
        try:
            data = np.load(filename)
            
            load_time = time.time() - start
            print(f"Loaded {len(data)} frames in {load_time:.2f}s")
            return data
        
        except Exception as e:
            print(f"Error loading from NPY: {e}")
            return None


    def batch_convert_mraw_to_hdf5(self, main_path, output_path, compression_level=1):
        """
        Batch convert MRAW files to HDF5 format
        
        Args:
            main_path (str): Directory containing MRAW files
            output_path (str): Directory to save HDF5 files
            compression_level (int): GZIP compression level (1-9)
            
        Returns:
            int: Number of files processed
        """
        processed_count = 0
        
        # Process all illumination directories
        for illumination in os.listdir(main_path):
            if 'illumination' not in illumination:
                continue
                
            print(f"Processing {illumination} directory")
            illumination_path = os.path.join(main_path, illumination)
            
            # Walk through directory structure
            for root, dirs, files in os.walk(illumination_path):
                # Create corresponding output directories
                if dirs and os.path.basename(root).startswith('FC'):
                    rel_path = os.path.relpath(root, main_path)
                    new_dir = os.path.join(output_path, rel_path)
                    
                    for dir_name in dirs:
                        os.makedirs(os.path.join(new_dir, dir_name), exist_ok=True)
                
                # Process CIHX files
                for file in files:
                    if not file.endswith(".cihx"):
                        continue
                        
                    input_file = os.path.join(root, file)
                    rel_path = os.path.relpath(os.path.dirname(input_file), main_path)
                    output_dir = os.path.join(output_path, rel_path)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    output_file = os.path.join(output_dir, file.replace('.cihx', '.h5'))
                    
                    # Skip if output file already exists
                    if os.path.exists(output_file):
                        print(f"Skipping {input_file} - output already exists")
                        continue
                    
                    # Load and save video
                    images, info = self.load_mraw_video(input_file)
                    if images is not None:
                        self.save_to_hdf5(images, output_file, compression_level)
                        processed_count += 1
                        
                        # Free memory
                        del images
                        gc.collect()
        
        return processed_count 