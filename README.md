Multi-Task Learning with Sentence Transformers
Overview
This repository contains the implementation of a multi-task learning framework using sentence transformers. The project is structured into four main tasks:

Sentence Transformer Implementation - Implementation of a sentence transformer model that encodes input sentences into fixed-length embeddings.
Multi-Task Learning Expansion - Expansion of the sentence transformer to handle multiple NLP tasks, specifically sentence classification and named entity recognition (NER).
Training Considerations - Discussion on various training scenarios including freezing different parts of the model during training.
Training Loop Implementation - Implementation of a training loop for the multi-task learning model, with detailed explanations on handling data, the forward pass, and metrics.


Repository Contents
task_1.py - Code for Task 1 implementation.
task_2.py - Code for Task 2 implementation.
task_3.py - Theoretical discussions for Task 3.
task_4.py - Training loop code for Task 4.
Dockerfile - Docker configuration file to build the environment.
requirements.txt - List of Python packages required for the project.


Docker Usage
Pulling the Docker Image
To pull the pre-built Docker image from Docker Hub:

docker pull sherifbakr1/assessment:latest

To run the Docker container:

docker run -it --rm sherifbakr1/assessment:latest .


Using the Application
Once inside the container, you can run the Python scripts for each task:


python task_1.py
python task_2.py
python task_3.py
python task_4.py

Building the Docker Image
If you want to rebuild the Docker image with any changes you make:

Clone the repository:

git clone url
cd directory
Build the Docker image:


docker build -t ghcr.io/sherifbakr1/assessment:latest .

For any questions or further information, please contact sherif.bakr@du.edu
