"""
This file is used to draw and save the structural diagram of the model.
The main purpose of this code is to load a pre-trained Keras model and use the plot_model function to plot the structure of the model.
First, it ensures that the path to Graphviz is added to the environment variable, this is because the plot_model function uses Graphviz internally to plot the model structure. The code then builds the full path to the model file from the configured model directory path and loads the model. Next, it checks to see if a folder named 'media' exists, and if it doesn't, creates it.
Finally, it constructs the full path to the output file and calls the plot_model function to plot and save the structure diagram of the model to the specified file.
"""

# Import the required libraries
import os  # For handling OS functionalities, such as file paths
import keras  # For loading and manipulating Keras models
from keras.utils import plot_model  # For plotting Keras models

from config import MODEL_DIR_PATH  # Get the model directory path from the config file

# Construct the full path for the model file
model_path = os.path.join(MODEL_DIR_PATH, 'Emotion_Voice_Detection_Model.h5')

# Load the model
restored_keras_model = keras.models.load_model(model_path)

# Check if the 'media' directory exists, if not, create it
media_directory = 'media'  # Or a specific path, depending on where the 'media' directory is located
if not os.path.exists(media_directory):
    os.makedirs(media_directory)

# Construct the full path for the output file
output_file_path = os.path.join(media_directory, 'model_project.png')

# Plot and save the model structure
plot_model(restored_keras_model, to_file=output_file_path)

