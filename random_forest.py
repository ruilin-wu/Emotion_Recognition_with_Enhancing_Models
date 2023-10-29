# Import the required libraries and modules
import joblib  # For loading and saving Python objects
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from sklearn.metrics import classification_report  # For displaying a text report of the main classification metrics

from config import SAVE_DIR_PATH  # Import the path from the config file

# Define a class for training the model
class TrainModel:

    @staticmethod
    def train_random_forest(X, y) -> str:
        """
        A static method to train the model using Random Forest.
        """

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # Create an instance of the Random Forest classifier
        rforest = RandomForestClassifier()
        # Fit the model using the training data
        rforest.fit(X_train, y_train)
        # Predict on the test set
        predictions = rforest.predict(X_test)

        # Print the classification report
        print(classification_report(y_test, predictions))
        return "Completed"  # Return a string indicating training completion

# The following code will execute when this script is run directly
if __name__ == '__main__':
    print('Training started')  # Print a message indicating the start of training
    # Load the feature set and label set from files
    X = joblib.load(SAVE_DIR_PATH + '\\X.joblib')
    y = joblib.load(SAVE_DIR_PATH + '\\y.joblib')
    # Call the method to train the Random Forest model
    RANDOM_FOREST = TrainModel.train_random_forest(X=X, y=y)
