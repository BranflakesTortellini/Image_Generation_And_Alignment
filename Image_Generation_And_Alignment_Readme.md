****************************************************************
Image Generator SCRIPT
****************************************************************
The Image Generator script allows you to generate a set of grayscale images with customizable properties such as line direction, line spacing, square size, and image quality. It can generate images with horizontal lines, vertical lines, or a combination of both.

How to Use
Run the script in a Python environment.
The GUI (Graphical User Interface) will open.
Select an output folder where the generated images will be saved.
Specify the resolution of the images, square size, line width, line spacing, and number of images to generate.
Choose the line direction preference: horizontal, vertical, or 50/50 (equal number of horizontal and vertical images).
Select the image quality: normal, high, or low.
Click the "Generate Images" button to start the image generation process.
The progress bar will show the progress of image generation.
Once the process is complete, a message box will appear confirming the completion.
The generated images will be saved in the specified output folder, with filenames "image_0.png", "image_1.png", and so on.
An additional image, "image_ideal.png", representing the ideal image based on the selected line direction, will also be saved.
Dependencies
The script requires the following dependencies:

OpenCV (cv2)
NumPy (numpy)
tkinter (tkinter)
tqdm (tqdm)
Make sure to install these dependencies before running the script.

Note
The script uses a GUI for input selection, so make sure you have a GUI environment available (e.g., running the script in a desktop environment or a Jupyter Notebook with GUI support).
That's it! You can now generate a set of grayscale images with customizable line properties using the Image Generator script.


****************************************************************
Image Alignment Script
****************************************************************
This script allows you to align a set of images based on their unique features. It utilizes computer vision techniques and machine learning to automatically calculate the alignment parameters (dx and dy) needed to align the images.

Features
Extraction of Unique Features: The script uses the SIFT (Scale-Invariant Feature Transform) algorithm to extract unique features from the images. These features serve as distinctive markers for alignment.

Feature Matching: The extracted features are then matched between the test images and an ideal image using a brute-force matching algorithm. This ensures that corresponding features are found for alignment.

Regression Model: A regression machine learning model is trained using the extracted features and alignment parameters (dx and dy) of a set of training images. This model learns the relationship between the features and the required alignment parameters.

Alignment Process: During the alignment process, the script estimates the alignment parameters for each test image based on its unique features and the ideal image. It uses the trained regression model to calculate the alignment parameters.

Visualization: The aligned images are displayed and saved, showing the unique features and the alignment paths from the test images to the corresponding features in the ideal image.

Requirements
Python 3.x
OpenCV
TensorFlow
scikit-learn
Plotly
Usage
Prepare your input images: Ensure that you have a set of images to align, including an ideal image (marked as "_ideal" in the filename) and other test images.

Run the script: Launch the script and select the input folder containing the images. The script will automatically extract the unique features, train the regression model, and align the images.

View the results: The aligned images will be saved, and a training report will be generated with alignment distances and success rate. You can also visualize the training and validation plot to assess the model's performance.

Experiment and fine-tune: You can continue training on an existing model, adjust the model architecture, or try different feature extraction algorithms to improve alignment accuracy.

Note: This script utilizes a regression model, which is a type of machine learning algorithm that learns to predict numerical values (alignment parameters) based on input features (extracted image features). It combines computer vision techniques with machine learning to achieve image alignment based on unique features.

Please ensure that you have the necessary dependencies installed and refer to the documentation for more detailed instructions on usage and customization.

Enjoy aligning your images with ease using this script!
