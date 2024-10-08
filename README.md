# Machine-Translation-using-NLP-and-Deep-Learning

This project implements a machine translation system that translates sentences from one language to another using advanced deep learning techniques, specifically focusing on Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, Gated Recurrent Units (GRU), and the Attention Mechanism. The implementation is done in a Jupyter Notebook (machine_translation.ipynb), leveraging popular machine learning frameworks such as TensorFlow and Keras.

# Table of Contents
  - **Introduction**
  - **Dataset**
  - **Model Architecture**
    - **Encoder-Decoder Architecture**
    - **Recurrent Neural Networks (RNN)**
    - **Long Short-Term Memory (LSTM)**
    - **Gated Recurrent Unit (GRU)**
    - **Attention Mechanism**
  - **Project Structure**
  - **Dependencies**
  - **License**

# Introduction
  Machine translation is a challenging task in Natural Language Processing (NLP) that involves automatically translating text from one language to another. In this project, we build a machine translation model that uses sequence-to-sequence learning (Seq2Seq) architecture with the attention mechanism to improve translation accuracy.

  The key components in this project include:

  Encoder-Decoder Architecture: A model that converts the source sentence into a fixed-length vector (encoder) and then converts this vector back into a target sentence (decoder).
  RNNs, GRUs, LSTMs: Recurrent neural networks that model the sequential nature of languages.
  Attention Mechanism: A mechanism that allows the decoder to focus on specific parts of the source sentence during translation.
  Model Architecture
  Encoder-Decoder Architecture
  The encoder-decoder architecture is commonly used in sequence-to-sequence tasks like machine translation. Here's how it works:

  The encoder processes the input sentence and generates a hidden representation or context vector, which summarizes the information of the entire input sequence.
  The decoder takes this context vector and generates the target sequence one word at a time.
  In this project, we use RNN-based architectures (LSTMs, GRUs) for both the encoder and the decoder to handle the sequential nature of text data.

# Dataset
  The dataset used in this project consists of paired Spanish and English sentences. The dataset was preprocessed to clean the text, remove punctuation, and tokenize the sentences for feeding into the model.

# Recurrent Neural Networks (RNN)
  Recurrent Neural Networks (RNNs) are a class of neural networks that are particularly suited for sequential data such as text. Unlike traditional feedforward neural networks, RNNs maintain a hidden state that can "remember" information from previous time steps, making them powerful for tasks like language modeling and machine translation.
  
  However, RNNs suffer from issues like the vanishing gradient problem, which makes it difficult for them to learn long-term dependencies in sequences. To address this, we use advanced variants of RNNs, namely LSTM and GRU.

# Long Short-Term Memory (LSTM)
  Long Short-Term Memory (LSTM) networks are a special type of RNN that can learn long-term dependencies more effectively by introducing a gating mechanism. The LSTM unit has three gates:
  
  Forget Gate: Decides which information should be thrown away or kept.
  Input Gate: Updates the cell state with new input information.
  Output Gate: Controls the output based on the cell state.
  LSTMs are ideal for handling the long-range dependencies in sequences that are crucial for machine translation tasks.

# Gated Recurrent Unit (GRU)
  Gated Recurrent Units (GRUs) are another variant of RNNs that aim to address the vanishing gradient problem. GRUs simplify the LSTM architecture by using only two gates:
  
  Update Gate: Decides how much of the previous information to retain.
  Reset Gate: Controls how much of the past information should be forgotten.
  GRUs are computationally more efficient than LSTMs while providing comparable performance, making them a popular choice for sequence-based tasks.

# Attention Mechanism
  The Attention Mechanism is an enhancement to the traditional encoder-decoder architecture. The problem with using only a fixed-length context vector (as in vanilla Seq2Seq models) is that it forces the model to condense all the information of the input sequence into a single vector. This leads to a loss of information, especially for long sequences.
  
  Attention helps the decoder "attend" to different parts of the input sequence at each step of the decoding process. This allows the model to focus on the most relevant parts of the input when predicting each word in the target sentence, which leads to better translation performance.
  
  In this project, we implement the attention mechanism to improve the quality of translations, especially for long sentences.

# Project Structure
  The project includes the following files:
  
 - **machine_translation.ipynb**: This is the main Jupyter notebook that contains the code for preprocessing the dataset, defining the model, training the model, and evaluating it.
  - **requirements.txt**: A file that contains the list of Python packages and dependencies required to run the project.
  - **.gitignore**: A file to specify which files/folders to ignore when pushing the project to GitHub.
  - **dataset**: The file spa.txt is the dataset used in this project.

# Key Steps in the Project
  - **Data Preprocessing**: The source and target language sentences are tokenized and padded to ensure uniform length. We use libraries like nltk and TensorFlow for this step.
  - **Model Definition**: We define an encoder-decoder architecture with attention using Keras's high-level APIs. The encoder processes the input sequence, and the decoder generates the output sequence, guided by the attention mechanism.
  - **Training**: The model is trained using pairs of input (source language) and output (target language) sentences, with categorical cross-entropy as the loss function and Adam as the optimizer.
  - **Evaluation**: The trained model is evaluated using BLEU scores, a popular metric for assessing the quality of machine translation systems.
  
# Prerequisites
  Python 3.x
  Jupyter Notebook

# Dependencies
  The project requires the following Python packages, listed in requirements.txt:
  
  tensorflow
  keras
  numpy
  pandas
  nltk
  matplotlib
  sklearn

# License
  GNU General Public License v3.0
  
  Copyright (C) 2024 Vishweshwar Reddy Veerannagari
  
  Everyone is permitted to copy and distribute verbatim copies
  of this license document, but changing it is not allowed.
  
  Preamble
  
  The GNU General Public License is a free, copyleft license for software and other kinds of works.
  
  The licenses for most software and other practical works are designed to take away your freedom to share and change the works. By contrast, the GNU General Public License is intended to guarantee your freedom to share and change all versions of a program--to make sure it remains free software for all its users. We, the Free Software Foundation, use the GNU General Public License for most of our software; it applies also to any other work released this way by its authors. You can apply it to your programs, too.
