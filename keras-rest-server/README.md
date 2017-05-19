## Keras-rest-server: A simple rest implementation for loading and serving keras models
---------------------------------------------------------------------------------------
## About:
This repository contains a very simple server implemented in flask which loads 
the Resnet model trained using Keras from its h5 weights.

## Getting started:
### Clone this repository
```
git clone https://github.com/ashavish/keras-rest-server-custom.git
cd keras-rest-server
sudo pip install -r requirements.txt
```
------------------

### Run the server (defaults to http://localhost:7171)
```
python server.py
```

### Send a post request to this server to test your model
```
python client.py
```
Code Adapted from : https://github.com/ansrivas/keras-rest-server