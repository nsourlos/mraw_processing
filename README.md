# mraw_read
Read mraw files
# MRAW File Reader and Video Processor

This repository contains code to read MRAW video files from high-speed cameras and process them into individual frames and video.

## Features

- Reads MRAW/CIHX files using pyMRAW
- Processes 12-bit images 
- Performs image enhancement:
  - Contrast adjustment
  - Brightness adjustment
  - Image rotation
- Saves individual frames as PNG files
- Creates an MP4 video from selected frames

## Requirements

- Python 3.x
- pyMRAW
- OpenCV (cv2)
- tqdm
- os

## Usage

1. Set the input path to your CIHX file
2. Specify the output path for frames and video
3. Run the script

This will read the MRAW file, process the images, and save the frames and video as specified.

## Notes

- The image enhancement steps are customizable, and you can modify them as needed for your specific application.
