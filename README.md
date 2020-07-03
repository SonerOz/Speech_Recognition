# Speech Recognition
## Overview

This is a simple speech recognition Flask app trained on the top of Keras API. The trained model **(model.h5)** takes an audio file [Launching the Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) as an input and predict the class of speech from bird, cat, dog, happy, wow, house, denomination.

## Motivation

Recent years, speech recognition systems have gotten into real-world applications all over. From cars, to healthcare (especially people with disabilities), from military applications to daily education life. And that let me put some effort into learning this technology and its applications to improve my skills and encourage others to learn. 

## Technical Aspect

This project is divided into two part:
1. Prepare the data and train it with deep learning model using Keras.
2. Build a Flask app and set up NGINX and uWSGI then dockerize it and upload to AWS EC2. 

## Installation

The Code is written in Python 3.7. To install the required packages and libraries, run this command in the project directory after cloning the repository:

> pip install -r requirements.txt

## Directory Tree

├── local 

    │   ├── classifier

    │           ├── data.json

    │           ├── prepare_dataset.py

    │           ├── train.py

    │   ├──dataset 

    │   ├── test

    │   ├── client.py

├── server

    │   ├── flask

    │      ├── app.ini

    │      ├── .dockerignore

    │      ├── Dockerfile

    │      ├── keyword_spotting_service.py

    │      ├── model.h5

    │      ├── server.py

├── nginx

    │   ├── Dockerfile

    │   ├── nginx.conf

├── docker-compose.yml

├── init.sh

├── README.md

└── requirements.txt

## Technologies Used
![Logo](https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg)
![Logo](https://keras.io/img/logo.png)
![Logo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSC-jZly0k4laGREHIjjPhniafRjSgf1Kmathj-AeVT31Z1y4g&s)
![Logo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTEER3XSOlsOmjI9nUd-QrvUzl9WRuEI-fWm8ukSZ3SDAGbEMFL&s)
![Logo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRemHo0Nc895r3xB23rRIBtvmICT6F1cNbB00GsBtD56CRWqlYe&s)
![Logo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSazzLRzzN-ZBk2BNv-y1YEqu6Q5sKrGePjhlfBJG0f_kOLc_CQ&s)
![Logo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTQ3Zn5HwL_aSKJdWRV59_VYgKIXr3EJUxHhnsaenMFixyvLo4&s)