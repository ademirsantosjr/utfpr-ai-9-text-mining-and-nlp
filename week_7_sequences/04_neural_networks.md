# Deep Learning Architecture for Natural Language Processing

With the sequences of embeddings in hand, the final step is to feed this data into a deep neural network. Multiple layers of neural networks are used to interpret the relationships between tokens and extract the necessary patterns to perform tasks such as text classification, sentiment analysis, among others.

This combination of layers forms a **Deep Learning** architecture.

## Architecture Overview

The data processing flow in our model can be summarized as follows:

```
   Raw Text
       |
       v
+-----------------+
| Vectorization   | --> Generates sequences of indices from the text.
| Layer           |
+-----------------+
       |
       v
+-----------------+
| Embedding       | --> Converts sequences of indices into sequences of vectors (embeddings).
| Layer           |
+-----------------+
       |
       v
+-----------------+
| Neural Network  | --> Analyze the sequences of vectors to learn patterns.
| Layers (CNN,    |
| LSTM, etc.)     |
+-----------------+
       |
       v
   Final Output
(Ex: Classification)
```

## The Role of Neural Network Layers

The neural network layers are the core of learning. They receive the sequence of vectors from the embedding layer and learn to identify complex patterns. Depending on the architecture used, they can capture everything from the relationship between neighboring words to contexts that extend throughout the entire sentence.

### Types of Layers for Sequence Analysis

Two layer architectures are especially effective for processing text sequences:

#### 1. Convolutional Neural Networks (CNNs)

Although they are best known for their success in computer vision, CNNs can be adapted to work with sequential data (in one dimension, `Conv1D`).

- **How it works:** The layer applies a "kernel" (or filter), which acts as a **sliding window** that moves along the sequence of embeddings.
- **What it learns:** At each position, the kernel analyzes a small group of neighboring words (similar to an n-gram). This allows the network to learn to detect very significant **local patterns**, such as the negation "did not like" or expressions like "very good".

#### 2. Long Short-Term Memory (LSTM)

LSTMs are a type of **Recurrent Neural Network (RNN)**, specifically designed to handle sequential data.

- **How it works:** Unlike a traditional neural network, an LSTM has an internal "loop". The output of one time step is used as input for the next step, allowing information to persist throughout the sequence.
- **What it learns:** The LSTM has a sophisticated mechanism of "gates" that allows the network to decide what to **remember** and what to **forget** from the information it has already processed. This makes it extremely powerful for capturing **long-term dependencies** in the text — for example, connecting a pronoun at the end of a paragraph to the subject mentioned at the beginning.

## Main Model Hyperparameters

When building this architecture, three input hyperparameters are fundamental to defining the model's behavior:

1.  **Vocabulary Size:** Limits the number of unique tokens that the model will consider. A larger vocabulary captures more words, but increases the complexity of the model.

2.  **Embedding Dimension:** Defines the size of the embedding vector for each token. Larger dimensions can capture richer semantic relationships, but require more data and computational power.

3.  **Sequence Length:** Determines the fixed size of the input sequences (after padding or truncation). It is a trade-off decision between maintaining more information from the text and the computational cost.
