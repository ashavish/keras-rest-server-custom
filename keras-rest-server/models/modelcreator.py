from keras.models import model_from_json
from keras.models import load_model

try:
    import cPickle as pickle
except:
    import pickle

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

import base64
import numpy as np
from io import BytesIO

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image as kerasimage
import requests




class ModelOperations(object):
    """
    """

    def __init__(self):
        pass

    def load_model(self, model_path):
        try:

            model = load_model(model_path)
            return model
        except:
            raise Exception('Failed to load model')


    def save_model(self, model, json_path, weights_path):
        """
        Helper wrapper over savemodels and saveweights to help keras dump
        the weights and configuration
        """
        json_string = model.to_json()
        with open(json_path, 'w') as f:
            f.write(json_string)
        model.save_weights(weights_path)


class Predictor(object):
    """
    """

    def __init__(self, model_path,
                 **kwargs):

        modoperations = ModelOperations()
        self.model = modoperations.load_model(model_path)
    
    
    def preprocess(self,X_input):
        # Loading Image If X_input is base64 encoded form of resized image
        # X_input = base64.decodestring(X_input)
        # Use either :
        # with open('/home/kicompute/keras-rest-server/test.jpeg','wb') as f:
        #      f.write(X_input) 
        # Or
        # image = BytesIO(base64.b64decode(X_input))
        # img = kerasimage.load_img(image)
        #
        ##############
        # Downloading / Loading Image Directly if X_input is URL
        # path = "/tmp/downloadedpics/test.jpeg"
        # resp = requests.get(X_input,stream=True)
        # if resp.status_code == 200:
        #     with open(path, 'wb') as f:
        #         for chunk in resp:
        #             f.write(chunk)
        # img = kerasimage.load_img(path,target_size=(224, 224))        
        ###############
        # img = kerasimage.load_img(X_input,target_size=(224, 224))
        # x = kerasimage.img_to_array(img)
        x = np.array(X_input)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def predict(self, X_input):
        """
        Make predictions, given some input data
        This normalizes the predictions based on the real normalization
        parameters and then generates a prediction

        :param X_input
            input vector to for prediction
        """   
        X_input = self.preprocess(X_input)
        x_pred = self.model.predict(X_input)
        return x_pred


