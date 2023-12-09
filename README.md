# Jetson-AI-Specialist

Project Documentation for Jetson AI Specialist

## Overview

The project aims to harness artificial intelligence as one of the pillars of Industry 4.0 to enhance efficiency, reduce errors, and provide complete inventory visibility, thereby contributing to smarter and more agile storage management.

The application of AI in product warehouses enables us to:

* Detect and recognize product stock

* Real-time updates on product availability

* Integration into existing systems

The project is focused on the storage station of a modular production system (MPS-500), where artificial vision is implemented to obtain real-time stock information from the station and manage this information in other processes.

It has two AI models:

1. The first model will be responsible for detecting the pieces.
2. The second model identifies the color of the piece.

In addition to this, a segmentation algorithm was implemented to obtain and manage the information.

![Diseño sin título](https://github.com/alejo-jose/Jetson-AI-Specialist/assets/67164878/a64e74c0-adf6-4bdd-92e2-c46560a0a235)

## Installation
The versions of the libraries used are:
* Torch - 1.12.0
* OpenCV - 4.7.0
* Numpy - 1.20.3
* Keras - 2.4.3
* Tensorflow 2.4.1

Additionally, a NVIDIA Jetson Nano 4 GB and a webcam are required.

## Usage

The program runs the AI models on photos acquired from a camera housed in the storage station; the prediction results can be observed in resulting images that are saved. 
Subsequently, the information extracted from the photos is stored in a matrix to process that information.








