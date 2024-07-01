import json
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import uvicorn

from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from image_processor import process_image_as_pil
from PIL import Image
from pydantic import BaseModel
from torchvision import models

class FeatureExtractor(nn.Module):
    """
    A Class for a feature extraction model. Has the same architecture as resnet50.
    That means it takes in an image in tensor form and returns 1000 outouts.
    """
    def __init__(self, decoder: dict = None):
        super(FeatureExtractor, self).__init__()
        
        # Load up resnet50
        self.resnet50 = models.resnet50(weights='IMAGENET1K_V1')
        
        # Set the architecture of the model to be the same as that of resnet50
        self.layers = torch.nn.Sequential(self.resnet50)

        self.decoder = decoder

    def forward(self, image):
        """
        The forward pass of the model
        """
        X = image.float()
        X = self.layers(X)
        return X

    def predict(self, image):
        """
        Same as the forward method, but doesn't keep track of grad to improve compute time
        """
        with torch.no_grad():
            x = self.layers(image)
            return x

# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str



try:
    # Set the correct device 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the decoder pickle
    with open("Pickle_Files/decoder_pickle", 'rb') as decoder_pickle:
        decoder = pickle.load(decoder_pickle)
    
    # Create the model
    feature_extraction_model = FeatureExtractor(decoder)

    # Load the model to device
    feature_extraction_model.to(device)

    # Load the parameters
    model_parameters = torch.load('Model_Parameters/image_model.pt', map_location=torch.device(device))
    feature_extraction_model.load_state_dict(model_parameters)
    print('Model parameters succesfully loaded!')
    pass
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location.")

try:
    # Load up the faiss seach index from `faiss_pickle`.
    with open("Pickle_Files/faiss_pickle", 'rb') as pickly:
        index = pickle.load(pickly)
    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location.")


app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
    msg = "API is up and running!"
    return {"message": msg}

  
@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):

    pil_image = Image.open(image.file)

    # Returns a toch tensors from the PIL Image
    image_tensor = process_image_as_pil(pil_image)

    # Stores the tensor in the device
    image_tensor = image_tensor.to(device)

    # Extract the image_embeddings from the feature extraction model
    feature_embeddings = feature_extraction_model(image_tensor)

    # Stores feature embedding tensor as cpu and removes grad attribute so we can cast it as a numpy array
    feature_embeddings = feature_embeddings.to('cpu')
    feature_embeddings = feature_embeddings.detach().numpy()

    # Finally cast feature embeddings as a numpy array
    feature_embeddings = np.array(feature_embeddings)

    feature_embeddings = feature_embeddings.tolist()

    return JSONResponse(content={
    "features": feature_embeddings, # Return the image embeddings here
    })
  
@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)

    # Extract the image embedding from the FASTApi JSON Response obtained from `predict_image`:
    response = predict_image(image)
    response = response.body
    response = json.loads(response)
    embeddings = response['features']

    # Cast the image embedding as a numpy array of type `float32`.
    embeddings = np.array(embeddings)
    embeddings = embeddings.astype('float32')

    # Use the FAISS algorithm to retrieve the indicies of the closest images.
    nearest_vectors = index.search(embeddings, 3)[1]

    # Turns the FAISS results to a normal python list, so that it can be parsed by JSON.
    nearest_vectors = nearest_vectors.tolist()

    return JSONResponse(content={
    "similar_index": nearest_vectors, # Return the index of similar images here
    })

if __name__ == '__main__':
    uvicorn.run('api:app', host="0.0.0.0", port=8080)
