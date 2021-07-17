# Simple Keras REST API

This repository contains an example using Keras pre-trained models as service.

I'm using Flask to create an REST API to make predictions.

## Getting started

This repository has a `.devcontainer`, so you can use a DevContainer with [Visual Studio Code](https://code.visualstudio.com/).

## Starting Keras server

Use Makefile or launch `python main.py`

`make server.run`

## Submiting requests to the keras server

Requests can be submittted via:

1. CURL

`curl -X POST -F image=@data/lynx.jpg 'http://localhost:5000/api/predict'

> Replace `data/lynx.jpg` with the path to your image.

2. Simple Request python file

I've created a file `simple_request.py`, you can run to make a simple HTTP/POST request.

You can launch in your terminal

- `make playground`
- `python simple_request.py`
