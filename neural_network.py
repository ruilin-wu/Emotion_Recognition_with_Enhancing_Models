"""
Neural network train file.
This code defines a class called TrainModel, which contains a static method train_neural_network,
Used to train a neural network for emotion recognition.
This method first splits the data set into a training set and a test set.
Then build a convolutional neural network (CNN),
The network includes convolutional layers, dropout layers, flatten layers and fully connected layers.
The model uses 'sparse_categorical_crossentropy' as the loss function,
and use the RMSprop optimizer. During training, both loss and accuracy are recorded and plotted.
Finally, the performance of the model is evaluated through the confusion matrix and classification report, and the trained model is saved to the specified path.
In the if __name__ == '__main__': section, the code loads the feature and label data,
And call the train_neural_network method to train the network.
"""
# Import the required libraries and functions
import os  # For handling operating system functionalities, such as file paths
import joblib  # For loading and saving Python objects
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting and visualization
import seaborn as sns  # For better visualization of the confusion matrix
from keras.layers import Dense, Conv1D, Flatten, Dropout, Activation, MaxPooling1D, GRU  # For building neural network layers
from keras.models import Sequential  # For creating a sequential model
from sklearn.metrics import confusion_matrix, classification_report  # For evaluating model performance
from sklearn.model_selection import train_test_split  # For splitting the dataset
from keras.layers import LSTM
from config import SAVE_DIR_PATH  # Get the save directory path from the config file
from config import MODEL_DIR_PATH  # Get the model directory path from the config file
from config import MEDIA_SAVE_PATH  # Get the model directory path from the config file
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import RMSprop

# Define a class to train the model
class TrainModel:

    @staticmethod
    def train_neural_network(X, y) -> None:
        """
        This function trains a neural network.
        """

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Expand the data dimensions for the neural network
        x_traincnn = np.expand_dims(X_train, axis=2)
        x_testcnn = np.expand_dims(X_test, axis=2)

        # Print the shapes of the training and testing data
        print(x_traincnn.shape, x_testcnn.shape)
        
        '''
        # Original CNN       
        model = Sequential()       
        model.add(Conv1D(64, 5, padding='same', input_shape=(40, 1)))       
        model.add(Activation('relu'))       
        model.add(Dropout(0.2))       
        model.add(Flatten())        
        model.add(Dense(8))       
        model.add(Activation('softmax'))
        '''
        
        
        model = Sequential()
        # Input shape is (number of training files, 40, 1)
        # First round of 1D CNN
        model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', input_shape=(40, 1)))
        model.add(Dropout(0.2))  # 20% Dropout
        model.add(MaxPooling1D(pool_size=2))
        # Run 1D CNN again with a different kernel size
        model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
        model.add(Dropout(0.2))  # 20% Dropout
        # Flatten the output
        model.add(Flatten())
        # Final Dense layer with softmax activation function
        model.add(Dense(units=8, activation='softmax')) 
           
        '''
        # GRU
        model = Sequential()       
        model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', input_shape=(40, 1)))
        model.add(Dropout(0.2))  
        model.add(MaxPooling1D(pool_size=2))       
        model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
        model.add(Dropout(0.2))         
        model.add(GRU(128, return_sequences=False))  
        model.add(Dense(units=8, activation='softmax'))  
        '''
        
        '''
        #LSTM
        model = Sequential()
        model.add(LSTM(128, input_shape=(40, 1), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(8))
        model.add(Activation('softmax'))
        '''
        # Print the model summary
        print('model summary')
        print(model.summary())

        # Compile the model
        lr = 0.001  # Define the learning rate
        optimizer = RMSprop(learning_rate=lr)

        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Train the model
        cnn_history = model.fit(x_traincnn, y_train, batch_size=16, epochs=150, validation_data=(x_testcnn, y_test))

        # Plot the loss graph
        plt.plot(cnn_history.history['loss'])
        plt.plot(cnn_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # Save the loss image
        loss_png_path = os.path.join(MEDIA_SAVE_PATH, "loss.png")
        plt.savefig(loss_png_path)
        plt.close()  # Close the image

        # Plot the accuracy graph
        plt.plot(cnn_history.history['accuracy'])
        plt.plot(cnn_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # Save the accuracy image
        accuracy_png_path = os.path.join(MEDIA_SAVE_PATH, "accuracy.png")
        plt.savefig(accuracy_png_path)

        # Save val_accuracy to a file
        val_accuracy_path = os.path.join(MEDIA_SAVE_PATH, "val_accuracy.txt")
        with open(val_accuracy_path, 'a') as file:  # Use 'a' mode so that data is appended each time, not overwritten
            for value in cnn_history.history['val_accuracy']:
                file.write(str(value) + '\n')

        # Make predictions using the model
        predictions = model.predict(x_testcnn)
        predictions = np.argmax(predictions, axis=1)  # Find the index with the maximum probability
        new_y_test = y_test.astype(int)
        matrix = confusion_matrix(new_y_test, predictions)

        # Print the classification report and confusion matrix
        report = classification_report(new_y_test, predictions)
        print('classification_report')
        print(report)
        print('confusion_matrix')
        print(matrix)

        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}
        # Get the emotion labels
        emotion_labels = [label_conversion[key] for key in sorted(label_conversion.keys())]
        # Visualize the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.yticks(rotation=0)  # Rotate the y-axis labels horizontally
        matrix_png_path = os.path.join(MEDIA_SAVE_PATH, "confusion_matrix.png")
        plt.savefig(matrix_png_path)
        plt.close()

        # Visualize the classification report
        plt.figure(figsize=(6, 6))
        plt.text(0.1, 1.25, 'Classification Report:', {'fontsize': 12}, fontweight='bold')
        plt.text(0.1, 1, report, {'fontsize': 10})
        plt.axis('off')
        report_png_path = os.path.join(MEDIA_SAVE_PATH, "classification_report.png")
        plt.savefig(report_png_path)
        plt.close()

        model_name = 'Emotion_Voice_Detection_Model.h5'
        # Save the model and weights
        if not os.path.isdir(MODEL_DIR_PATH):
            os.makedirs(MODEL_DIR_PATH)
        model_path = os.path.join(MODEL_DIR_PATH, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

# The following code will execute when this script is run directly
if __name__ == '__main__':
    print('Training started')  # Print a message indicating the start of the training
    # Load the features and labels
    X = joblib.load(SAVE_DIR_PATH + '\\X_MFCC_song.joblib')
    y = joblib.load(SAVE_DIR_PATH + '\\y_MFCC_song.joblib')
    # Start training the neural network
    NEURAL_NET = TrainModel.train_neural_network(X=X, y=y)

