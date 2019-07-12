# RTHN: A RNN-Transformer Hierarchical Network for Emotion Cause Extraction
This repository contains the code for our IJCAI 2019 paper: 

Rui Xia, Mengran Zhang, and Zixiang Ding. RTHN: A RNN-Transformer Hierarchical Network for Emotion Cause Extraction. IJCAI 2019. https://arxiv.org/abs/1906.01236

Please cite our paper if you use this code.

## Abstract
The emotion cause extraction (ECE) task aims at discovering the potential causes behind a certain emotion expression in a document. Techniques including rule-based methods, traditional machine learning methods and deep neural networks have been proposed to solve this task. However, most of the previous work considered ECE as a set of independent clause classification problems and ignored the relations between multiple clauses in a document. In this work, we propose a joint emotion cause extraction framework, named RNNTransformer Hierarchical Network (RTHN), to encode and classify multiple clauses synchronously. RTHN is composed of a lower word-level encoder based on RNNs to encode multiple words in each clause, and an upper clause-level encoder based on Transformer to learn the correlation between multiple clauses in a document. We furthermore propose ways to encode the relative position and global predication information into Transformer that can capture the causality between clauses and make RTHN more efficient. We finally achieve the best performance among 12 compared systems and improve the F1 score of the state-of-the-art from 72.69% to 76.77%.

## Dependencies
Python 3

Tensorflow 1.8.0

## Usage
python prepare_data.py

python RTHN.py


