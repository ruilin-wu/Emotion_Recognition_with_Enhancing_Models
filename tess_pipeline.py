"""
This file builds 2 additional actor folders (25 and 26) using features from the
Toronto emotional speech set (TESS) dataset: https://tspace.library.utoronto.ca/handle/1807/24487

These stimuli were modeled on the Northwestern University Auditory Test No. 6 (NU-6; Tillman & Carhart, 1966).
A set of 200 target words were spoken in the carrier phrase "Say the word _____'
by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions
(anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 stimuli in total.
Two actresses were recruited from the Toronto area. Both actresses speak English as their first language,
are university educated, and have musical training. Audiometric testing indicated that
both actresses have thresholds within the normal range.

Authors: Kate Dupuis, M. Kathleen Pichora-Fuller

University of Toronto, Psychology Department, 2010.

TESS data can be downloaded from here: https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess/data

To facilitate the feature creation, the TESS data have been renamed using the same naming convention adopted
by the RAVDESS dataset explained below:

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

In case of TESS files, an example below. We do not care of assigning values other than the ones
specified below as those are not used by the model, hence we are assigning random integers.
- 03 (Random)
- 01 (Random)
- 01 (This varies according to the fact in TESS we have 1 emotion less then RAVDESS: calm).
- 01 (Random)
- 03 (Random).
- 01 (Random)
- 01 (Random. I thought initially to put 25 if YAF, 26 if OAF, but that is not needed as the pipeline is not
using the actor information from the filename, only the mfccs extracted from librosa and the target emotion).


The main function of this code is to process the audio files of the TESS (Toronto emotional speech set) data set, rename and copy them to a new directory to be consistent with the file naming method of the RAVDESS data set, thereby facilitating subsequent emotions. Recognition model training.

Specifically, the code does the following:

1. **Class definition**: The `TESSPipeline` class is defined, which contains a static method `create_tess_folders`, which is used to handle the renaming and copying of files.

2. **File processing**:
     - Loop through all files in the TESS dataset.
     - Determine whether the file name starts with 'OAF' (representing an audio type).
     - Based on the sentiment information in the file name (such as 'happy', 'sad', etc.), use the predefined `label_conversion` dictionary to convert it into the corresponding code (such as '03' for 'happy').
     - Generate new filenames that contain random numbers but follow a specific format and contain converted sentiment codes.
     - Copy the original file to a new location and use a new file name.

3. **Path configuration**: Import path variables from the `config` file, such as `TRAINING_FILES_PATH` and `TESS_ORIGINAL_FOLDER_PATH`. These variables point to the storage location of the data set and the target location of the processed file.

4. **Script Execution**: If this script is run directly (rather than imported as a module), the `__main__` section will call the `create_tess_folders` method and pass in the path of the TESS dataset, thus triggering the above file processing process.

Overall, the purpose of this script is to automate the processing of audio files so that they conform to a specific naming and organizational structure in preparation for subsequent data loading and model training steps.

"""


# Import the operating system interface module
import os
# Import the advanced file operations module
import shutil
# Import the random number module
import random

# Import the training file path from the config file
from config import TRAINING_FILES_PATH
# Import the TESS original folder path from the config file
from config import TESS_ORIGINAL_FOLDER_PATH

# Define the TESSPipeline class
class TESSPipeline:

    # Define a static method to create tess folders
    @staticmethod
    def create_tess_folders(path):
        """
        This comment explains the function and background of this method
        """
        # Initialize the counter
        counter = 0

        # Define the label conversion dictionary
        label_conversion = {'01': 'neutral',
                            '03': 'happy',
                            '04': 'sad',
                            '05': 'angry',
                            '06': 'fear',
                            '07': 'disgust',
                            '08': 'ps'}

        # Traverse all files in the specified path
        for subdir, dirs, files in os.walk(path):
            for filename in files:
                # If the filename starts with 'OAF'
                if filename.startswith('OAF'):
                    # Define the destination path
                    destination_path = os.path.join(TRAINING_FILES_PATH, 'Actor_26')  # This considers the operating system's path separator

                    # Get the full path of the old file
                    old_file_path = os.path.join(os.path.abspath(subdir), filename)

                    # Separate the filename and extension
                    base, extension = os.path.splitext(filename)

                    # Traverse the label conversion dictionary
                    for key, value in label_conversion.items():
                        # If the filename ends with a certain emotion
                        if base.endswith(value):
                            # Generate a list of random numbers
                            random_list = random.sample(range(10, 99), 7)
                            # Convert the random number list to a string and join with '-'
                            file_name = '-'.join([str(i) for i in random_list])
                            # Create a new filename containing the correct emotion label
                            file_name_with_correct_emotion = file_name[:6] + key + file_name[8:] + extension
                            # Define the full path of the new file
                            new_file_path = os.path.join(destination_path, file_name_with_correct_emotion)
                            # Copy the file to the new location
                            shutil.copy(old_file_path, new_file_path)

                else:
                    # Define the destination path
                    destination_path = os.path.join(TRAINING_FILES_PATH, 'Actor_25')  # This considers the operating system's path separator

                    # Get the full path of the old file
                    old_file_path = os.path.join(os.path.abspath(subdir), filename)

                    # Separate the filename and extension
                    base, extension = os.path.splitext(filename)

                    # Traverse the label conversion dictionary
                    for key, value in label_conversion.items():
                        # If the filename ends with a certain emotion
                        if base.endswith(value):
                            # Generate a list of random numbers
                            random_list = random.sample(range(10, 99), 7)
                            # Convert the random number list to a string and join with '-'
                            file_name = '-'.join([str(i) for i in random_list])
                            # Create a new filename containing the correct emotion label
                            file_name_with_correct_emotion = (file_name[:6] + key + file_name[8:] + extension).strip()
                            # Define the full path of the new file
                            new_file_path = os.path.join(destination_path, file_name_with_correct_emotion)
                            # Copy the file to the new location
                            shutil.copy(old_file_path, new_file_path)

# If the current script is the main program, execute the create_tess_folders method
if __name__ == '__main__':
    TESSPipeline.create_tess_folders(TESS_ORIGINAL_FOLDER_PATH)
