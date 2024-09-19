# Human Activity Recognition of WiFi Signals Based on Stacked Group Convolutional Sparse Attention

This repository provides a deep learning framework specifically designed for identifying human activities through WiFi signals. It supplements our research paper titled "Human Activity Recognition Based on Probabilistic Sparse Attention and Enhanced Group Convolutional with WiFi Signals" by providing base line methods in a series of comparative experiments. 

Our reference code is adpated from repository [WiFi CSI Sensing Benchmark](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark). It supports the optional models ResNet18, ResNet50, GRU, BiLSTM and ViT for three public datasets: NTU Fi HAR, NTU Fi HumanID, and UT_HAR_data. 

To run the code, simply execute "Python run. py -- model [model name] -- dataset [dataset name]", for example: "Python run. py -- model BiLSTM -- dataset NTU Fi_HAR". The orignal dataset and source code can be found in [WiFi CSI Sensing Benchmark](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark).
