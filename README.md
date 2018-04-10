# Sales forecasting with Keras and Tensorflow

The contents will allow you to play with a Keras model forecasting sales for the data from: https://www.kaggle.com/c/rossmann-store-sales 

You will be able to run your code locally and on google cloud platform for scalability.

## Description

This beginner-level workshop will guide you through implementing a sales forecasting neural network model based on an openly available dataset. We will go through some basic neural network theory and apply a systematic model improvement approach to get from a simple model to a more complex one, trying out various tricks along the way.

## Contents
- *data*: train and store coming from kaggle Rossman case: 
  - *train1, eval1*: data filtered for just one store (date ascending)
- *output* - where the checkpoint models are stored
- *gcp-output* - where the checkpoint models from the cloud are downloaded
- *notebooks* - useful notebooks
- *scripts* - scripts (mainly gcloud) for dealing with google cloud
- *trainer* - main model

## Requirements

- python: beginner level
- a laptop with some software configured (see setup instructions)
- pandas, numpy, sklearn: nice-to-have

## Setup instructions

You will need: 

- a clone of: https://github.com/gregaw/sales-forecasting-with-nn.git
- anaconda (https://conda.io/docs/user-guide/install/index.html)
- [optional] google cloud ml-engine (there's free tier available) and shell (https://cloud.google.com/sdk/)

Once you get those installed (and have this git repo cloned locally), run:

> conda create -n mlengine python=2.7 anaconda

> source activate mlengine

> pip install -r requirements.txt

Note  : (you can also set it up with python=3.6, but you might have problems running this on GCP ML-engine)
Note2 : check the setup.txt with some dumps of the environment correctly setup.

## Running locally
Use run_next_local.py, which will create a new job_name (with sequential numbers)

## Running on GCP ML-engine
Use scripts under *scripts*
