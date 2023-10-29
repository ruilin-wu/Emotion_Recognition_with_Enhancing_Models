"""
Configuration file: using this one to keep all the paths in one place for various imports.

TRAINING_FILES_PATH = Path of the training files. Here there are

- the RAVDESS dataset files (Folders Actor_01 to Actor_24
- the TESS dataset renamed files (Folders Actor_25 and Actor_26)

SAVE_DIR_PATH = Path of the joblib features created with create_features.py

MODEL_DIR_PATH = Path for the keras model created with neural_network.py

TESS_ORIGINAL_FOLDER_PATH = Path for the TESS dataset original folder (used by tess_pipeline.py)

"""
# Import the pathlib library for object-oriented filesystem path operations
import pathlib
# Import the os library for OS-level operations, such as handling file paths
import os

# Get the absolute path of the current working directory
working_dir_path = pathlib.Path().absolute()

# Define the path for training files, which are located in the 'features' folder
TRAINING_FILES_PATH = os.path.join(str(working_dir_path), 'features')
# Define the path to save joblib features, which are located in the 'joblib_features' folder
SAVE_DIR_PATH = os.path.join(str(working_dir_path), 'joblib_features')
# Define the path to save the model, which will be saved in the 'model' folder
MODEL_DIR_PATH = os.path.join(str(working_dir_path), 'model')
# Define the path for the original TESS dataset folder, located in the 'TESS_Toronto_emotional_speech_set_data' folder
TESS_ORIGINAL_FOLDER_PATH = os.path.join(str(working_dir_path), 'TESS_Toronto_emotional_speech_set_data')
# Define a path for the 'examples' folder, though not specifically mentioned in this code, it might be used to store example inputs or outputs
EXAMPLES_PATH = os.path.join(str(working_dir_path), 'examples')
# Define a path for the 'media' folder
MEDIA_SAVE_PATH = os.path.join(str(working_dir_path), 'media')
