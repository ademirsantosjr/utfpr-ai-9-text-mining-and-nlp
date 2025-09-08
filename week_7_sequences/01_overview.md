# Overview: Sequences, Embeddings and Deep Learning in NLP

In this section, we explore an advanced approach to document classification, overcoming the limitations of the traditional *Bag of Words (BoW)* model. Instead of just considering the presence or absence of words, we now analyze the **sequence** of terms, capturing the context and structure of the language more effectively.

The combination of **sequences**, **word embeddings**, and **deep neural networks (Deep Learning)** allows us to create sophisticated language models, which are the foundation for technologies like ChatGPT and Google Gemini.

## Pillars of the Approach

Our methodology is based on three fundamental concepts:

### 1. Sequences
We abandon the *Bag of Words* model to work with sequences. Instead of a disordered "bag of words," we represent texts as numerical vectors where each number corresponds to a token (word or sub-word) in a vocabulary, preserving the original order of the information.

### 2. Embeddings
For each token in a sequence, we use *embeddings* to generate a dense vector representation that captures its **semantic meaning**. Unlike a simple numerical representation, embeddings place words with similar meanings close to each other in the vector space, enriching the model with a deeper understanding of the language.

### 3. Deep Learning
We use deep learning models, such as Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs), to process the sequences of embeddings. These architectures are capable of learning complex patterns and long-term dependencies in textual data, making them ideal for tasks such as:
- Document classification
- Text generation
- Chatbot systems
- Sentiment analysis

## Technologies Used

For the practical implementation of these concepts, we will use the following Python libraries:

- **Gensim:** A powerful tool for creating and manipulating *word embeddings*.
- **Keras (TensorFlow):** A high-level framework for building and training our Deep Learning models, as well as assisting us in transforming texts into sequences.
