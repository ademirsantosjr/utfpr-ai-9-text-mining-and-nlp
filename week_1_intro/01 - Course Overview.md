# Overview of the Text Mining and Natural Language Processing Course

## Definition of Natural Language Processing (NLP)

Natural Language Processing (NLP) is a subfield of artificial intelligence focused on the interaction between computers and human language. The main technology companies define NLP as follows:

*   **Google:** Uses machine learning to unravel the structure and meaning of texts, allowing the extraction of information about people, places, and events.
*   **Amazon:** It is a machine learning technology that enables computers to interpret, manipulate, and understand human language.
*   **IBM:** Combines computational linguistics with statistical and machine learning models so that computers can recognize, understand, and generate text and speech.

## NLP Applications

NLP applications are vast and cover several domains, such as:

*   Sentiment analysis on social networks and e-commerce comments.
*   Text classification and generation.
*   Content extraction from documents.
*   Information retrieval systems (search engines).
*   Automatic summarization and translation.

## The Role of Machine Learning in NLP

Machine learning is a central pillar of NLP. Different approaches are used, including:

*   Supervised Learning
*   Unsupervised Learning
*   Deep Learning

## The Data Mining Process in NLP

NLP applications follow a data mining process that involves two main stages:

### 1. Pre-processing

In this phase, the natural language text is structured through tasks such as:

*   **Tokenization:** Division of the text into smaller units (tokens or words).
*   **Part-of-Speech (POS) Tagging:** Definition of the syntactic function of each word (verb, noun, etc.).
*   **Named Entity Recognition (NER):** Identification of names of people, organizations, locations, etc.

### 2. Feature Extraction

The objective of this stage is to transform the text into a format that machine learning algorithms can understand. The main approaches studied will be:

*   **Bag of Words:** A model that represents the text as a set of its words, disregarding grammar and order.
*   **Sequences and Embeddings:** More advanced techniques that capture the context and meaning of words, used in deep neural networks.

## Course Tools and Libraries

To implement the NLP tasks, we will use the following libraries:

*   **Pre-processing:**
    *   **spaCy:** For tokenization, POS tagging, and NER.
    *   **NLTK:** For specific tasks like *stemming*.
*   **Feature Extraction and Modeling:**
    *   **Scikit-Learn:** For the Bag of Words model and machine learning algorithms.
    *   **Gensim, Keras, and Hugging Face:** To work with embeddings and deep learning.

## Course Structure

The discipline will be divided into three main parts:

1.  **Part 1 (Week 1):** Focus on pre-processing tasks with the **spaCy** library.
2.  **Part 2 (Weeks 2-5):** Work with the **Bag of Words** model, addressing document classification, dimensionality reduction, and topic discovery.
3.  **Part 3 (Weeks 6-8):** Exploration of models based on **Deep Learning**, including Word Embeddings, text generation, and the use of the **Transformers** library (extra content).
