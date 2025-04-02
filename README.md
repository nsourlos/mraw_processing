# MRAW Video Processor

A Python toolkit for processing MRAW (cihx) files from high-speed cameras, with specialized processing for both incident and back illumination video data. Supports both traditional OpenCV-based and PyTorch-based processing methods.

## Features

- Convert MRAW (.cihx) files to compressed HDF5 format for efficient storage and faster loading
- Process high-speed camera footage with:
  - Traditional OpenCV-based processing for:
    - Incident illumination videos
    - Back illumination videos
  - PyTorch-based processing with customizable kernels:
    - Edge detection
    - Gaussian filtering
    - Sobel operators
    - Custom kernel support
- Automatic illumination type detection based on folder names
- Analyze wave heights and track wave propagation using column analysis
- Generate visualizations and animations of wave dynamics

## Installation

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/nsourlos/mraw_read.git
   cd mraw_read
   ```

2. Create a new conda environment and activate it:
   ```bash
   conda create -n video_analysis python==3.10 -y
   conda activate video_analysis
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
mraw_read/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── main.py           # Main script for running the pipeline
│   ├── mraw_loader.py     # MRAW file loading and HDF5 conversion
│   ├── image_processor.py # Traditional and PyTorch image processing
│   ├── torch_processor.py # PyTorch-specific kernels and operations
│   └── column_analyzer.py # Wave height analysis
├── data/                  # Directory for input data
│   ├── subfolder_back/     # Any folder with 'back' in the name is processed as back illumination
│   └── subfolder_incident/ # Any folder with 'incident' in the name is processed as incident illumination
├── output/                # Directory for output results
└── README.md             # Documentation
```

## Usage

### Data Organization

Organize your data files using a naming convention that indicates the illumination type:
```
data/
├── any_name_with_back_in_it/      # Processed as back illumination
│   ├── file1.cihx
│   ├── file2.npy
│   └── ...
├── back_illumination_test015/     # Processed as back illumination
│   └── ...
├── incident_illumination_sample/  # Processed as incident illumination
│   └── ...
└── any_name_with_incident_in_it/  # Processed as incident illumination
    └── ...
```

The system automatically detects the illumination type from the folder name:
- If the folder name contains "back", files are processed with back illumination methods
- If the folder name contains "incident", files are processed with incident illumination methods 
- If neither is found, it defaults to incident illumination

### Running the Pipeline

```bash
python src/main.py
```

The pipeline will:
1. Scan all subfolders in the `data` directory
2. Detect illumination type based on folder names (looking for "back" or "incident")
3. Process each file according to its detected illumination type
4. Convert MRAW files to HDF5 format automatically
5. Apply both traditional and PyTorch processing methods
6. Analyze wave heights and create visualizations
7. Track wave roller positions

### Available Kernels

The toolkit includes several predefined kernels for PyTorch-based processing:

- Edge Detection
- Sobel Operators (Top, Bottom, 5x5 variants)
- Gaussian Smoothing
- Mean Filter
- Emboss
- Sharpen

## Processing Pipeline

### Incident Illumination Processing

For incident illumination videos, the processing pipeline:
1. Rotates and normalizes frames
2. Applies CLAHE for better contrast
3. Calculates difference between background and wave frames
4. Processes using either:
   - Traditional method: Noise removal and thresholding
   - PyTorch method: User-specified kernel operations
5. Enhances visibility using CLAHE

### Back Illumination Processing

For back illumination videos, the pipeline:
1. Applies adaptive gamma correction
2. Uses different contrast for bright and dark areas
3. Applies gain for HDR-like effect
4. Calculates difference between background and wave frames
5. Processes using either:
   - Traditional method: Noise removal and thresholding
   - PyTorch method: User-specified kernel operations
6. Enhances visibility using CLAHE

## Output Files

The pipeline organizes output by folder and file name:
```
output/
├── [folder_name]_[file_name]/    # For each input file
│   ├── traditional/              # Traditional processing results
│   │   ├── frames/               # Processed frames
│   │   └── combined/             # Combined visualizations
│   ├── pytorch/                  # PyTorch processing results
│   │   ├── frames/               # Processed frames 
│   │   ├── visualizations/       # Processing step visualizations
│   │   └── combined/             # Combined visualizations
│   └── analysis/                 # Analysis results
│       ├── wave_heights_plot.png # Wave height plots
│       ├── wave_heights.npy      # Raw wave height data
│       └── wave_height_animation.mp4 # Animation
└── [folder_name]_[file_name].h5  # Converted HDF5 files
```

## Performance Considerations

- Processing large videos may require significant memory
- Use compression when saving to HDF5 to balance storage space and loading speed
- Consider using frame ranges (start_idx, end_idx) for testing and development

## License

[MIT License](LICENSE)

## Acknowledgments

This project was developed for processing high-speed camera footage of dam break wave experiments.

## Contact

For questions or support, please contact [your-email@example.com]. 