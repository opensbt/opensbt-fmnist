from tensorflow import keras
from keras.layers import Input
import numpy as np
from fmnist.models.Model1_fmnist import Model1_fmnist

from mnist.config import EXPECTED_LABEL, num_classes

class Predictor:

    # Load the pre-trained model.
    input_tensor = Input(shape=(28,28,1))
    model = Model1_fmnist(input_tensor=input_tensor)
    print("Loaded model from disk")
    
    @staticmethod
    def predict(img, explabel = EXPECTED_LABEL):
        explabel = (np.expand_dims(explabel, 0))

        # Convert class vectors to binary class matrices
        explabel = keras.utils.to_categorical(explabel, num_classes)

        # print(f"Explabel {explabel}")
        explabel = np.argmax(explabel.squeeze())

        # Predictions vector
        predictions = Predictor.model.predict(img)

        prediction1, prediction2 = np.argsort(-predictions[0])[:2]

        # print(f"Predicted vs expected: {prediction1} / {explabel}")
        # print(f"EXPECTED_LABEL: {EXPECTED_LABEL}")

        
        # Activation level corresponding to the expected class
        confidence_expclass = predictions[0][explabel]

        if prediction1 != explabel:
            confidence_notclass = predictions[0][prediction1]
        else:
            confidence_notclass = predictions[0][prediction2]

        confidence = confidence_expclass - confidence_notclass

        return prediction1, confidence

    ''' Return the list of all predictions (softmax layer values)'''
    @staticmethod
    def predict_extended(img, explabel = EXPECTED_LABEL):
        explabel = (np.expand_dims(explabel, 0))

        # Convert class vectors to binary class matrices
        explabel = keras.utils.to_categorical(explabel, num_classes)
        explabel = np.argmax(explabel.squeeze())

         #Predictions vector
        predictions = Predictor.model.predict(img)

        # print(f"Predictions {predictions}")

        
        return predictions[0]
