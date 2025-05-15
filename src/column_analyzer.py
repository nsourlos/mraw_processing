import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

class column_analyzer():
    def setup_columns(self, image_width, image_height, num_columns=5, bin_start=100, bin_end=200):
        """
        Setup column positions based on image dimensions
        
        Args:
            image_width (int): Width of the image
            image_height (int): Height of the image
            num_columns (int): Number of columns to analyze
            bin_start (int): Start 1st column after this many pixels
            bin_end (int): End with the nth column that half of this many pixels before the end of img
            
        Returns:
            list: Positions of analysis columns
        """
        # Calculate equally spaced column positions as in the notebook
        column_positions = [bin_start + i * (image_width - bin_end) // (num_columns-1) for i in range(num_columns)]
        print(f"Image dimensions: {image_width} x {image_height}")
        print(f"Column positions: {column_positions}")
        
        return column_positions

    def analyze_columns(self, frames_dir, num_columns=5, output_dir=None, max_frames=None, 
                    skip_top_pixels=190, middle_wave_height=500, threshold_level=10,
                    column_width=1, bin_start=100, bin_end=200):
        """
        Analyze all frames in directory to extract wave heights using upper contour of the wave
        
        Args:
            frames_dir (str): Directory containing processed frames
            num_columns (int): Number of columns to analyze
            output_dir (str, optional): Directory to save analysis results
            max_frames (int, optional): Maximum number of frames to process
            skip_top_pixels (int): Number of pixels to skip from the top
            middle_wave_height (int): Used to get only the upper contour of the wave
            threshold_level (int): Threshold level for binarization (0-255)
            column_width (int): Width of each column in pixels
            bin_start (int): Start 1st column after this many pixels
            bin_end (int): End with the nth column this many pixels before the end
            
        Returns:
            tuple: (bin_arrays, column_positions, frame_count)
        """
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get list of frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.jpg')])
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        frame_count = len(frame_files)
        print(f"Found {frame_count} frames to analyze")
        
        # Initialize the first frame to setup columns
        first_frame_path = os.path.join(frames_dir, frame_files[0])
        print(f"First frame: {first_frame_path}")
        first_frame = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)
        image_height, image_width = first_frame.shape[:2] 
        # print(f"Width, height: {image_width} {image_height}")
        
        # Calculate column positions
        column_positions = self.setup_columns(image_width, image_height, num_columns, bin_start, bin_end)
        # print(f"Column positions: {column_positions}")
        
        # Initialize arrays to store height measurements for each column
        bin_arrays = [[] for _ in range(num_columns)]
        
        # Middle frame for visualization
        middle_frame_idx = frame_count // 2
        
        # Process each frame
        for i, frame_file in enumerate(tqdm(frame_files, desc="Getting columns")):
            frame_path = os.path.join(frames_dir, frame_file)
            # print(frame_path)
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply threshold to binarize the image
            _, thresh = cv2.threshold(frame, threshold_level, 255, cv2.THRESH_BINARY)
            
            # Process each column
            for j, col_center in enumerate(column_positions):
                # Extract column - Based on width starting and ending point is decided
                col_start = max(0, col_center - column_width//2)
                col_end = min(image_width, col_center + column_width//2 + (column_width % 2))  # Handle odd widths
                column = thresh[skip_top_pixels:, col_start:col_end]
                
                # Find highest white pixel (first white pixel from top) - wave's upper contour
                white_pixels = np.where(column > 0)[0]
                if len(white_pixels) > 0:
                    highest_point = white_pixels[0] + skip_top_pixels  # First white pixel from top
                else:
                    highest_point = image_height  # If no white pixels found, use bottom of image
                
                # Ensure we're getting the upper contour of the wave
                if image_height - highest_point < middle_wave_height:
                    highest_point = middle_wave_height
                
                # Store distance from bottom (bottom till upper part of the wave)
                bin_arrays[j].append(int(image_height - highest_point))
            
            # Create visualization for middle frame
            if i == middle_frame_idx and output_dir:
                # Convert threshold image to BGR so we can draw colored rectangles
                thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                img_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                # Draw a green rectangle for each column position
                for col_center in column_positions:
                    col_start = max(0, col_center - column_width//2)
                    col_end = min(image_width, col_center + column_width//2 + (column_width % 2))
                    cv2.rectangle(thresh_bgr,
                                (col_start, 0),  # top-left point
                                (col_end, image_height),  # bottom-right point 
                                (0, 255, 0),  # BGR color (green)
                                2)  # Thickness
                    cv2.rectangle(img_bgr,
                                (col_start, 0),  # top-left point
                                (col_end, image_height),  # bottom-right point 
                                (0, 255, 0),  # BGR color (green)
                                2)  # Thickness
                
                # Save both images with columns overlaid
                output_path = os.path.join(output_dir, f"columns_frame_{middle_frame_idx:05d}_{column_width}_pixels_width.jpg")
                cv2.imwrite(output_path, thresh_bgr)
                output_path_orig = os.path.join(output_dir, f"columns_frame_original_{middle_frame_idx:05d}_{column_width}_pixels_width.jpg")
                cv2.imwrite(output_path_orig, img_bgr)
        
        # Create plots with the results
        if output_dir:
            self.create_plots(bin_arrays, column_positions, frame_count, output_dir, column_width)
            
            # Save bin_arrays for each column as npy files
            for i, col_pos in enumerate(column_positions):
                output_path = os.path.join(output_dir, f'bin_array_column_{col_pos}_pixels.npy')
                np.save(output_path, np.array(bin_arrays[i]))
        
        return bin_arrays, column_positions, frame_count

    def create_plots(self, bin_arrays, column_positions, frame_count, output_dir, column_width=1):
        """
        Create and save analysis plots as in the notebook
        
        Args:
            bin_arrays (list): List of arrays containing wave heights for each column
            column_positions (list): List of column positions
            frame_count (int): Number of frames analyzed
            output_dir (str): Directory to save plots
            column_width (int): Width of column used for analysis
        """
        # Turn off interactive mode to prevent figures from opening
        plt.ioff()
        
        # Create the plot
        plt.figure(figsize=(15, 5))
        frames = range(1, frame_count + 1)
        
        for i in range(len(column_positions)):
            plt.subplot(1, len(column_positions), i+1)
            plt.plot(frames, bin_arrays[i])
            plt.title(f'Column at x={column_positions[i]} pixels')
            plt.xlabel('Frame')
            plt.ylabel('Height (pixels)')
            plt.grid(True)  # Add grid for better readability
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"time_series_graph_{column_width}_pixels_width.png"), 
                    dpi=300, bbox_inches='tight')

    def create_animation(self, frames_dir, output_dir, bin_arrays, column_positions, frame_count, num_columns=5, fps=10, max_frames=None, column_width=1):
        """
        Create animation of wave height evolution
        
        Args:
            frames_dir (str): Directory containing processed frames
            output_dir (str): Directory to save animation
            num_columns (int): Number of columns to analyze
            fps (int): Frames per second for animation
            max_frames (int, optional): Maximum number of frames to include
            skip_top_pixels (int): Number of pixels to skip from the top
            middle_wave_height (int): Used to get only the upper contour of the wave
            threshold_level (int): Threshold level for binarization (0-255)
            column_width (int): Width of each column in pixels
            bin_start (int): Start 1st column after this many pixels
            bin_end (int): End with the nth column this many pixels before the end
        """
        
        # Get list of frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.jpg')])
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        # Convert bin_arrays to numpy array for easier handling
        bin_arrays_np = np.array([np.array(col) for col in bin_arrays]).T  # shape: (frames, columns)
        
        # Set up the figure without displaying
        plt.ioff()  # Turn off interactive mode
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Read first frame for dimensions
        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        
        # Initialize plots
        img_plot = ax1.imshow(first_frame_rgb)
        ax1.set_title('Processed Frame')
        
        # Plot wave height lines
        lines = []
        for i in range(num_columns):
            line, = ax2.plot([], [], label=f'Column at x={column_positions[i]}')
            lines.append(line)
        
        ax2.set_xlim(0, len(frame_files))
        ax2.set_ylim(np.min(bin_arrays_np), np.max(bin_arrays_np) * 1.1)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Height (pixels)')
        ax2.set_title('Wave Height Evolution')
        ax2.legend()
        ax2.grid(True)
        
        # Text for frame counter
        frame_text = ax1.text(10, 30, '', color='white', fontsize=12, 
                            bbox=dict(facecolor='black', alpha=0.7))
        
        def init():
            frame_text.set_text('')
            for line in lines:
                line.set_data([], [])
            return [img_plot, frame_text] + lines
        
        def update(frame_idx):
            # Update frame image
            frame_path = os.path.join(frames_dir, frame_files[frame_idx])
            frame = cv2.imread(frame_path)
            
            # Draw column positions on frame
            frame_vis = frame.copy()
            for col_pos in column_positions:
                col_start = max(0, col_pos - column_width//2)
                col_end = min(frame.shape[1], col_pos + column_width//2 + (column_width % 2))
                cv2.rectangle(frame_vis,
                            (col_start, 0),
                            (col_end, frame.shape[0]),
                            (0, 255, 0),
                            2)
            
            frame_rgb = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)
            img_plot.set_array(frame_rgb)
            
            # Update wave height plots - plot all frames up to current
            frames = np.arange(frame_idx + 1)
            for i, line in enumerate(lines):
                line.set_data(frames, bin_arrays_np[:frame_idx + 1, i])
            
            # Update frame counter
            frame_text.set_text(f'Frame: {frame_idx}')
            
            return [img_plot, frame_text] + lines
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=range(len(frame_files)),
                            init_func=init, blit=True, interval=1000/fps)
        
        # Save animation
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'wave_height_animation.mp4')
        print(f"Saving animation to {output_path}")
        
        # Use a specific writer with appropriate settings
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Wave Analysis Tool'), bitrate=500)

        with tqdm(total=len(frame_files), desc="Rendering", unit="frame") as pbar:
            anim.save(output_path, writer=writer, progress_callback=lambda i, n: pbar.update(1))
        
        # Clean up
        plt.close('all')  # Close any other open figures
        del anim

    def detect_roller_area(self, frame, threshold=10, illumination_type='back'):
        """
        Detect the roller area in a frame by counting nonzero pixels after thresholding
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            int: Area estimate (number of nonzero pixels)
        """

        _, thresh = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        return np.sum(thresh > 0)  # Count nonzero pixels as area estimate

    def get_wave_roller(self, frames_dir, output_dir=None, starting_frame=0, max_frames=None,
                    start_row=190, end_row_offset=400, within_y_pixels=50, close_points_x=30,
                    threshold_area=100000, threshold_img=10, illumination_type='back', save_arrays=False):
        """
        Detect and track the wave roller position in sequential frames
        
        Args:
            frames_dir (str): Directory containing processed frames
            output_dir (str, optional): Directory to save visualization
            starting_frame (int): Frame to start analysis from
            max_frames (int, optional): Maximum number of frames to process
            start_row (int): Cut the n pixels at the top of the image
            end_row_offset (int): Cut the n pixels at the bottom of the image
            within_y_pixels (int): Check if points are within n pixels in y
            close_points_x (int): Only connect consecutive points if they're close enough
            threshold_area (int): Area threshold to detect wave roller presence
            save_arrays (bool): Whether to save all_frames and all_contours as numpy arrays (default: False)
            
        Returns:
            tuple: (wave_roller_coords, starting_frame, end_frame)
        """
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Get frame files and count
        print("Input Frames Directory:", frames_dir)
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.jpg')])
        n_frames = len(frame_files)
        print("Total number of frames in it:", n_frames)
        if max_frames:
            n_frames = min(n_frames, starting_frame + max_frames)
        
        # Find first frame with area above threshold
        for frame_num in tqdm(range(starting_frame, n_frames), desc="Finding wave roller start"):
            # if "traditional" in frames_dir:
            frame_path = os.path.join(frames_dir, f"frame_{frame_num:05d}.jpg")
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE) 
            # else: # maybe not do this here? Cause this seems to be fine NOTE: An alternative would be to just set the starting
            #     frame_path = os.path.join(frames_dir, f"frame_{(frame_num):05d}.npz")
            #     frame_file = np.load(frame_path) 
            #     frame = (frame_file.f.arr_0) * 255 # This needs to have values 0-255
            if frame is None: 
                continue

            area = self.detect_roller_area(frame, threshold_img, illumination_type)
            if area > threshold_area:
                starting_frame = frame_num
                print(f"\nFirst frame with area above {threshold_area} pixels: Frame {starting_frame} with area {area}")
                break
            # else:
            #     print("area",area)
        
        # Initialize arrays to store frames and contours
        all_frames = []  # Store frames with visualizations
        all_contours = []  # Store contour visualizations
        wave_roller_coords = []
        prev_area = None
        area_change_threshold = 50000  # Adjust based on expected contour size change
        flag = 0
        end_frame = n_frames
        
        for frame_num in tqdm(range(starting_frame, n_frames), desc="Tracking wave roller"):
            frame_path = os.path.join(frames_dir, f"frame_{frame_num:05d}.jpg")
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            
            if frame is None:
                continue
            
            # Cut top and bottom regions
            gray = frame[start_row:-end_row_offset, :] if end_row_offset > 0 else frame[start_row:, :]
            
            # Apply thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY) if illumination_type=='back' else cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to clean noise
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # Check for a sudden spike in detected area
                if prev_area is not None and abs(area - prev_area) > area_change_threshold:
                    print(f"Warning: Sudden change in contour area at frame {frame_num}")
                    continue
                prev_area = area
                
                # Get bounding box and filter invalid contours
                x_min, y_min, width, height = cv2.boundingRect(largest_contour)
                
                if width > 0.999 * gray.shape[1] and flag == 0:
                    flag = 1
                    end_frame = frame_num
                    print(f"Wave roller out of frame in frame {frame_num}") #Last frame not saved

                    # Create video from frames at the end
                    video_path = os.path.join(output_dir, 'wave_roller_tracking.mp4')
                    frame_size = (frame.shape[1], frame.shape[0])
                    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, frame_size, True)
                    
                    # Read all debug frames from starting_frame to end_frame and write to video
                    for i in tqdm(range(starting_frame, end_frame ), desc="Creating video with wave roller..."):
                        debug_frame = cv2.imread(os.path.join(output_dir, f'debug_frame_{i:05d}.jpg'))
                        if debug_frame is not None:
                            out.write(debug_frame)
                    
                    out.release()
                    break
                
                elif flag == 0:  # Keep detecting contours
                    # Find the rightmost point
                    rightmost = tuple(largest_contour[largest_contour[:,:,0].argmax()][0])
                    x, y = rightmost
                    y += start_row  # Adjust for ROI offset
                    
                    # Create dictionary to store highest points for each x coordinate
                    highest_points = {}
                    for point in largest_contour:
                        px, py = point[0]
                        py_adjusted = py + start_row  # Adjust for ROI offset
                        
                        # Only consider points with x>5 and those that are above the wave roller
                        if px > 5 and py_adjusted < y and px < x:
                            # Update if this is the highest point for this x coordinate
                            if px not in highest_points or py < highest_points[px]:
                                highest_points[px] = py
                    
                    # Convert dictionary to filtered list of points with continuity check
                    filtered_points = []
                    prev_x = None
                    prev_y = None
                    
                    for px in sorted(highest_points.keys()):
                        curr_y = highest_points[px]
                        
                        if prev_x is not None:
                            # Check if points are within n pixels in y
                            if abs(curr_y - prev_y) > within_y_pixels:
                                continue
                            
                            # Add point only if it's continuous
                            filtered_points.append([[px, curr_y]])
                        else:
                            # First valid point
                            filtered_points.append([[px, curr_y]])
                        
                        prev_x = px
                        prev_y = curr_y
                    
                    # Create visualization frame
                    vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    # Create contour visualization frame (only if we're saving arrays)
                    contour_vis = np.zeros_like(vis_frame) if save_arrays else None
                    
                    # Draw the filtered points
                    for i in range(len(filtered_points)-1):
                        pt1 = tuple(filtered_points[i][0])
                        pt2 = tuple(filtered_points[i+1][0])
                        # Only connect points if they're close enough
                        if abs(pt2[0] - pt1[0]) <= close_points_x:
                            cv2.line(vis_frame[start_row:-end_row_offset if end_row_offset > 0 else start_row:, :], 
                                    pt1, pt2, (0, 255, 0), 2)
                            
                            # Only draw on contour_vis if we're saving arrays
                            if save_arrays and contour_vis is not None:
                                cv2.line(contour_vis[start_row:-end_row_offset if end_row_offset > 0 else start_row:, :], 
                                        pt1, pt2, (0, 255, 0), 2)
                    
                    # Mark the wave roller position
                    cv2.circle(vis_frame, (x, y), 5, (0, 0, 255), -1)
                    wave_roller_coords.append([x, y])
                    cv2.putText(vis_frame, f"Frame: {frame_num}, Pos: ({x}, {y})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Store frames if save_arrays is True
                    if save_arrays:
                        all_frames.append(vis_frame.copy())
                        if contour_vis is not None:
                            all_contours.append(contour_vis.copy())
                    
                    # Save visualization
                    if output_dir:
                        debug_path = os.path.join(output_dir, f'debug_frame_{frame_num:05d}.jpg')
                        cv2.putText(vis_frame, f"Area: {self.detect_roller_area(frame)}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                        cv2.imwrite(debug_path, vis_frame)
        
        print(f"Processed {len(wave_roller_coords)} frames with wave roller detection")
        
        # Save all_frames and all_contours as numpy arrays if requested
        if save_arrays and output_dir and len(all_frames) > 0:
            print("Saving frames and contours arrays...")
            np.save(os.path.join(output_dir, 'all_frames.npy'), np.array(all_frames))
            np.save(os.path.join(output_dir, 'all_contours.npy'), np.array(all_contours))
            print(f"Saved {len(all_frames)} frames and {len(all_contours)} contours as numpy arrays")

        return wave_roller_coords, starting_frame, end_frame