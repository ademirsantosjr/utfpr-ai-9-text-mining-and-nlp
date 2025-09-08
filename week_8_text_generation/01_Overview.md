
# Text Generation with Language Models

In this section, we will explore the concept of **Language Models** and how they can be applied to the task of **text generation**. Starting from a deep neural network architecture that we have already used for classification, we will make a fundamental adaptation so that the model, instead of classifying, learns to generate coherent text sequences.

## What are Language Models?

A **Language Model (LM)** is an artificial intelligence system trained to understand and generate text. In essence, it learns the probability distribution of word sequences in a given language.

The main objective of an LM is, given a context (a sequence of words), to predict the most likely next word. For example, for the sequence "The cat climbed on the", the language model could predict "roof" as the most likely continuation.

Important concepts in this area include:

- **Neural Language Models (NLM):** Models that use neural networks to learn the relationships between words.
- **Large Language Models (LLM):** Large-scale neural language models, trained with vast datasets, such as those that power applications like ChatGPT.

## From Classification to Generation: The Change in Architecture

In our previous exploration of sentiment analysis, we built Deep Learning models with the following architecture:

1. **Vectorization Layer:** Transformed the text into numerical sequences.
2. **Embedding Layer:** Created dense vector representations with semantic meaning for the tokens.
3. **Recurrent (LSTM) or Convolutional (CNN) Layers:** Learned the patterns in the sequences.
4. **Output Layer:** A `Dense` layer with a single neuron and sigmoid activation to predict a binary output (positive or negative sentiment).

For text generation, the main modification occurs in the output layer. We replace the dense layer of one neuron with a **`Dense` layer with `Softmax` activation function**.

- **Number of Neurons:** This new output layer will have a number of neurons equal to the size of our vocabulary (the number of unique words the model knows).
- **Model Output:** The output of the `Softmax` function is a probability distribution over the entire vocabulary. Each neuron will provide the probability of the corresponding word being the next word in the sequence.

## The Text Generation Process

Text generation will occur iteratively:

1. **Input (Seed):** We provide the model with an initial sequence of words (the "seed").
2. **Prediction:** The model processes the input and generates a probability distribution for the next word.
3. **Selection:** We identify the word with the highest probability.
4. **Concatenation:** We append the predicted word to the input sequence.
5. **Repetition:** The new sequence is used as input for the model to predict the next word, and the cycle continues, generating the text word by word.

## Our Project

In this project, we will train a language model with a specific corpus: **a set of regulations from the Federal University of Technology - Paraná (UTFPR)**. This means that our model will learn the syntax, vocabulary, and style of these documents, and the texts it generates will have a similar theme.

We will use libraries like Keras and spaCy to build, train, and evaluate our text generation model.
