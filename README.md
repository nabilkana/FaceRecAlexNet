## Facial Recognition using AlexNet and Cosine Similarity
This project implements a facial recognition system in MATLAB by training and evaluating multiple deep learning models based on AlexNet using the AT&T Face dataset . The best-performing models are aggregatedand , later extending it with cosine similarity for scalable face recognition—even for individuals the model wasn't originally trained on.

## Overview
Trained several CNN models using AlexNet as a base.

Aggregated the best model based on validation accuracy.

Removed the final classification layer to extract deep feature embeddings.

Used cosine similarity to measure closeness between face embeddings for recognition.

## Files
rochis.mat: code for evaluating performance .

trainingalex2.m: Script for fine tuning the model .

trainedMdel2.m: fine tuned model .

Facial_Recognition_Presentation.pptx: Slides explaining the methodology, results, and insights.

## Requirements
MATLAB R2021b or later

Deep Learning Toolbox
Image Processing Toolbox
Pretrained AlexNet (alexnet command or via Add-On Explorer)
AT&T Face dataset