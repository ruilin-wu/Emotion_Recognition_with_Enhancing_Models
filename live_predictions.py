"""
This file can be used to try a live prediction.
The main purpose of this code is to use the pre-trained model H5 to perform emotion recognition on real-time audio files.
It loads a trained Keras model and uses it to predict emotions in the provided audio file.
The prediction is based on the MFCC (Mel Frequency Cepstrum Coefficient) features of the audio file.
"""
import os
import keras
import librosa
import numpy as np

from config import EXAMPLES_PATH
from config import MODEL_DIR_PATH


class LivePredictions:
    """
    Main class of the application.
    """

    def __init__(self, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.file = file  # Audio file
        self.path = os.path.join(MODEL_DIR_PATH, 'Emotion_Voice_Detection_Model.h5')
        self.loaded_model = keras.models.load_model(self.path)  # Load the model

    def make_predictions(self):
        """
        Method to process the files and create your features.
        """
        data, sampling_rate = librosa.load(self.file)  # Load the audio file
        # Calculate MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        print(f'mfccs.shape is', mfccs.shape)
        mfccs = np.expand_dims(mfccs, axis=1)  # Convert (40,) to (40, 1)
        x = np.expand_dims(mfccs, axis=0)  # Add batch_size dimension, becomes (1, 40, 1)
        predictions = self.loaded_model.predict(x)  # This will return an array containing the probability distribution
        predicted_class = np.argmax(predictions, axis=-1)  # Get the index of the class with the highest probability
        print("Prediction is", " ", self.convert_class_to_emotion(predicted_class))

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Method to convert the prediction (integer) to a human-readable string.
        """
        # Mapping between emotion labels and integers
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value  # Get the corresponding emotion label
        return label  # Return the emotion label

if __name__ == '__main__':
    # Instantiate the class and make live predictions
    live_prediction = LivePredictions(file=os.path.join(EXAMPLES_PATH, '03-02-01-01-01-01-11.wav'))# Neutral
    live_prediction.loaded_model.summary()  # Print model summary
    live_prediction.make_predictions()  # Make predictions
    
    live_prediction = LivePredictions(file=os.path.join(EXAMPLES_PATH, '03-02-04-01-01-01-21.wav'))# Sad
    live_prediction.make_predictions()  # Predict on another file
    
    live_prediction = LivePredictions(file=os.path.join(EXAMPLES_PATH, '03-02-05-02-01-02-12.wav'))# Angry
    live_prediction.make_predictions()  # Predict on another file
    
    live_prediction = LivePredictions(file=os.path.join(EXAMPLES_PATH, '03-02-06-02-02-01-15.wav'))# Feardul
    live_prediction.make_predictions()  # Predict on another file
