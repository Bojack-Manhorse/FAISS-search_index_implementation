# Facebook-Marketplace-s-Recommendation-Ranking-System

## Introduction

This project creates an API to implement a Facebook AI Similarity Search (Faiss) with the aim of grouping together similar images based on a dataset consisting of images from an online marketplace. It can be broken down into four parts.

### Dataset Processing

We start of with a dataset consisting of details an images from an online marketplace. We process the dataset so that we are left with the images of the items being sold and the category (in the context of an online trade) they belong to, e.g. a sofa would belong to 'Home & Garden' and a laptop would belong to 'Computer & Software'.

### Training a Neural Network to classify the image categories

Once we've processed the dataset, we create a neural network to classify the category the of the image. This is done via transfer learning, we take the pretained model resnet50 (https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) amd retrain the last few layers to be specific to our dataset.

### Implementing FAISS

Once we've trained a neural network model, we feed the predictions from the model into a FAISS search index. Given any image, this allows us to quickly calculate which images from our dataset are most similar to the given image.

### API and Docker

Finally, we create an API using FASTAPI to implement the FAISS seach, and create a Docker image to run the entire program.

## Directions

The repository already contains a trained model for feature extraction, specifically the file `app/Model_Parameters/image_model.pt`. Thus the easiest way to run the program would be via a docker container.

### Docker Hub Method

A docker image of the project can be found at:

```
https://hub.docker.com/repository/docker/bojackmanhorse/faiss_app
```

To download the docker image, run 

```
docker pull bojackmanhorse/faiss_app:v1.3
```

in a terminal. Then to run the docker image, run

```
docker run -p 8080:8080 -it bojackmanhorse/faiss_app:v1.3
```

You should see the followin in the terminal:

![Image](Readme_images/Screenshot%20from%202024-07-01%2015-34-57.png)

Once the API is up and running, navigate to:

```
http://localhost:8080/docs
```

In a browser to test the program.

### Docker build method

A dockerfile is contained within this repository. To run it, first clone the repository and navigate to the folder `app`. Then build the docker image via:

```
docker build . -t <tage_name>
```

Then run the docker image as before:

```
docker run -p 8080:8080 -it <tage_name>
```

And finally naviagate to the OpenAPI documentation to test out the program:

```
http://localhost:8080/docs
```

### Run non containerized

While this is not recommended, it is possible to run the program without using a docker container. Clone the repository and naviaget to the app folder, then install the requirements (which will take some time):

```
pip install -r requirements.txt
```

Then run the api via:

```
python app.py
```

Then the api will be running on port 8080 as in the previous cases.

## Project details

### Training the model

The methods for training the feature extration model can be found in `Sandbox.ipynb`. If you would like to retrain the model parameters, re-run all the cells in that notebook, afterwards the new model parameters will automatically be saved in the `app` directory. Then rebuild the docker image so that the new parameters are used in the api.

We initially started of with resnet50, which is a pretrained image classification model which outputs a vector with 1000 entries. Then we extend it by adding extra layers so that the resulting model only has 13 entries (13 is the number of categories of our image dataset). Here's a graph showing how the loss reduced as we trained our model:

![Image](Readme_images/Screenshot%20from%202024-07-01%2015-36-27.png)

Of course, we'd like our model to be specific to the dataset of online marketplace trades, so we further train the model using stochastic gradient descent using the images we have. But we do not need to train the whole model, we only train the last layer of resnet50, and the few extra leyers we added on.

### Creating the FAISS search index

Given the model, we then cut off the extra layers that we added, so now we have a model with the same architecture as resnet50 but with the final layer having weights tuned better for our dataset. We then use the predictions from the model (called the feature extraction model) to build a FAISS search index. This will tell us what images are most similar to each other based on thier predictions from the feature extraction model.

### API

Here's a picture of the API:

![Image](Readme_images/Screenshot%202024-07-01%20at%2015-52-05%20FastAPI%20-%20Swagger%20UI.png)