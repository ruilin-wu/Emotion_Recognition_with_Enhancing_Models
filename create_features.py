"""
This files creates the X and y features in joblib to be used by the predictive models.
This code defines a class called CreateFeatures,
This class contains a static method features_creator,
Used to extract features from audio files and save them as .joblib files.
This method iterates through all audio files in the specified directory,
Extract MFCC (Mel Frequency Cepstral Coefficients) features from each file using the librosa library,
These features and their corresponding labels are then saved in a list.
Upon completion, features and labels are converted to NumPy arrays and saved in a .joblib file in the specified directory.
In the if __name__ == '__main__': section,
The code calls the features_creator method and specifies the path to the audio file and the directory where the features are saved.
"""

# Import required libraries
import os  # For interacting with the operating system
import time  # For tracking time
import joblib  # For saving/loading Python objects
import librosa  # For audio analysis
import numpy as np  # For numerical computations

from config import SAVE_DIR_PATH  # Get the save directory path from the config file
from config import TRAINING_FILES_PATH  # Get the training files path from the config file

# Define a class to create features
class CreateFeatures:

    @staticmethod
    def features_creator(path, save_dir) -> str:
        """
        This function creates a dataset and saves the data and labels in two files,
        X.joblib and y.joblib, both located in the joblib_features folder.
        Using this method, you can persistently save your features and quickly train new machine learning models,
        without having to reload the features using this pipeline every time.
        """

        lst = []  # Initialize an empty list to store features

        start_time = time.time()  # Record the start time

        # Iterate over all files in the specified path
        for subdir, dirs, files in os.walk(path):
            for file in files:
                try:
                    # Load the audio file, extract MFCC features, and store the filename and features in a new array
                    X, sample_rate = librosa.load(os.path.join(subdir, file), res_type='kaiser_fast')
                    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)                    
                    '''
                    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                    
                    chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
                    
                    contrast = np.mean(librosa.feature.spectral_contrast(y=X, sr=sample_rate).T, axis=0)
                    
                    combined_features = np.hstack([mfccs, chroma, contrast])
                    '''
                    # The following command converts labels from 1-8 to 0-7, as our predictor needs to start from 0
                    file = int(file[7:8]) - 1
                    arr = mfccs, file
                    lst.append(arr)  # Add the array to the list
                # If the file is invalid, skip it
                except ValueError as err:
                    print(err)
                    continue

        # Print the time required to load the data
        print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

        # Create X and y: the zip function combines all the first elements and all the second elements into separate lists
        X, y = zip(*lst)

        # Convert the lists to arrays
        X, y = np.asarray(X), np.asarray(y)

        # Check the shape of the arrays
        print(X.shape, y.shape)

        # Prepare the names for the feature dump files
        X_name, y_name = 'X.joblib', 'y.joblib'

        # Save the features and labels to the specified directory
        joblib.dump(X, os.path.join(save_dir, X_name))
        joblib.dump(y, os.path.join(save_dir, y_name))

        return "Completed"  # Return completion status

# When this script is run directly, the following code will execute
if __name__ == '__main__':
    print('Routine started')  # Print the start message
    # Call the function to create features and specify the path
    FEATURES = CreateFeatures.features_creator(path=TRAINING_FILES_PATH, save_dir=SAVE_DIR_PATH)
    print('Routine completed.')  # Print the completion message
