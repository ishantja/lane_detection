# Lane Detection and Segmentation 
This Python program detects straight lanes in a video, classifies them into left and right lanes, and finally colors them according to the type of lane (dashed=red or solid=green). It uses computer vision techniques and the OpenCV library for processing and analyzing video frames. The program provides visual feedback in real-time, with up to five frames displayed side by side, including the input frame, edges, cropped lane region, Hough lines, and the final lane segmentation with classification colors.

<p align="center">
  <img src=output/lane_detection_gif.gif alt="animated" />
</p>

## Hough Transform
The Hough Line Transform is an image processing technique used to detect straight lines in an image. It works by transforming the image from (x, y) coordinates to (ρ, θ) space, where ρ is the distance from the origin to a point on the line, and θ is the angle between the line and the x-axis.

The process involves:
**Edge Detection**: Detect edges in the image, typically using methods like the Canny edge detector.
**Voting**: For each edge pixel, vote for possible (ρ, θ) values that could represent a line.
**Peak Detection**: Identify peaks in the accumulator array, which indicate the presence of lines.
**Line Extraction**: Convert the (ρ, θ) values of the detected peaks back to (x, y) coordinates to obtain the line's parameters.

The Hough Line Transform is robust to noise and can detect lines, even if they are broken or partially obscured. It is widely used in tasks like lane detection in autonomous vehicles and barcode recognition. However, it may require parameter tuning and is best suited for detecting straight lines in images.


## Program Overview
The code is organized into a few main components:

### LaneDetectorPipeline Class
This class encapsulates the processing and analysis of video frames. It performs the following tasks:

### Pre-processing
Resizing the input frame, converting it to grayscale, applying Gaussian blur, and detecting edges using the Canny edge detector.
Region of Interest Selection: Cropping the frame to focus on the region where the lanes are expected to be.
### Hough Line Detection
Applying the Hough Line Transform to identify lines in the cropped region.
### Line Classification
Classifying lines into left and right lanes based on their slopes, and coloring them according to whether they are dashed (red) or solid (green).
### Visualizer Class
This class handles the visualization of processed frames. It allows you to display up to five different frames simultaneously and save them as a video. The code is designed to provide real-time feedback on lane detection and segmentation.

### Main Function
The main function initializes the LaneDetectorPipeline and Visualizer classes, reads frames from a video file, and processes each frame using the pipeline. It displays the input frame, edges, cropped lane region, Hough lines, and the final lane segmentation with classification colors. Press 'q' to exit the program.

## Dependencies
Make sure you have the following dependencies installed:
- OpenCV (cv2)
- NumPy
- Matplotlib (for visualization)

## Usage
Adjust the scale variable in the main function to control the resizing of the frames for faster processing.

Update the video file path in the **video_path = 'data/whiteline.mp4'** in the main function to your desired video source.

Run the code, and the program will display and save the processed frames.

**Press 'q'** to exit the program.

## Saving the Output
If you want to save the generated frames as a video, set the **save parameter** to True in the main function when initializing the Visualizer class. The processed frames will be saved as 'lanes_detection_output.mp4' in the output directory.

## Customization
You can customize various parameters, such as the region of interest, Hough Line Transform parameters, or line classification criteria to better suit your specific use case. This algorithm can only detect straight lanes. Try the chellenge video that has curved lanes to see the limitation of this algorithm. I will be posting a new algorithm to handle the curved lanes along with turn prediction. 

This code is a starting point for lane detection and segmentation in videos. You can further enhance it and integrate it into larger projects, such as autonomous driving systems or lane-keeping assist systems.

Enjoy lane detection and segmentation with this code!