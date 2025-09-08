# Text Mining and NLP

This repository contains the materials for the Text Mining and Natural Language Processing discipline, part of the Artificial Intelligence Specialization course.

## Week 1: Introduction to NLP and spaCy

The first week of the course provides an introduction to the fundamental concepts of Natural Language Processing (NLP) and the `spaCy` library.

### Topics Covered:

*   **Course Overview:** Introduction to NLP, its applications, and the course structure.
*   **Introduction to spaCy:** An overview of the `spaCy` library and its architecture.
*   **Tokenization and POS Tagging:** Understanding how to split text into tokens and assign grammatical tags.
*   **Named Entity Recognition (NER):** Identifying and classifying named entities in text.
*   **Lemmatization and Stopwords:** Reducing words to their base form and removing common words.

### Files:

*   `week_1_intro/01 - Course Overview.md`: Course overview and introduction to NLP.
*   `week_1_intro/02 - Introduction to Spacy.md`: Introduction to the `spaCy` library.
*   `week_1_intro/03 - Tokenization and POS Tagger.md`: Notes on tokenization and Part-of-Speech tagging.
*   `week_1_intro/04 - Named Entity Recognition.md`: Notes on Named Entity Recognition.
*   `week_1_intro/05 - Lemmas and Stopwords.md`: Notes on lemmatization and stop words.
*   `week_1_intro/week1_text_mining_pln_spacy.ipynb`: Jupyter Notebook with practical examples of the concepts covered in week 1.

## Week 2: Text Representation with Bag-of-Words and TF-IDF

The second week focuses on vector text representation, exploring the Bag-of-Words (BOW) model and the TF-IDF metric.

### Topics Covered:

*   **The Bag-of-Words (BOW) Model:** Understanding the fundamentals of the BOW model.
*   **Practical Implementation with CountVectorizer:** How to implement BOW using Scikit-learn's `CountVectorizer`.
*   **A More Advanced Metric: TF-IDF:** Introduction to Term Frequency-Inverse Document Frequency.
*   **Practical Implementation with TfidfVectorizer:** How to implement TF-IDF using Scikit-learn's `TfidfVectorizer`.
*   **Applications in Information Retrieval and Recommendation:** Use cases for these text representation models.

### Files:

*   `week_2_bow/01 - The Bag-of-Words (BOW) Model.md`: Notes on the Bag-of-Words model.
*   `week_2_bow/02 - Practical Implementation with CountVectorizer.md`: Notes on `CountVectorizer`.
*   `week_2_bow/03 - A More Advanced Metric: TF-IDF.md`: Notes on TF-IDF.
*   `week_2_bow/04 - Practical Implementation with TfidfVectorizer.md`: Notes on `TfidfVectorizer`.
*   `week_2_bow/05 - Applications in Information Retrieval and Recommendation.md`: Notes on applications.
*   `week_2_bow/Week2_Text_Representation_From_BOW_to_TF-IDF.md`: Summary of the concepts covered in week 2.

## Week 3: Text Classification and Sentiment Analysis

The third week focuses on applying machine learning techniques to text classification, with a special emphasis on sentiment analysis.

### Topics Covered:

*   **Machine Learning Review:** A concise review of the fundamental concepts of machine learning.
*   **Supervised Machine Learning Concepts:** Detailed explanation of supervised learning concepts relevant to text classification.
*   **Training and Evaluation:** Guidelines and best practices for training and evaluating text classification models.
*   **Sentiment Analysis:** An in-depth exploration of sentiment analysis, including its applications and the architecture of sentiment classifiers.
*   **Implementation:** Step-by-step implementation of a sentiment classifier using Bag-of-Words and machine learning algorithms.

### Files:

*   `week_3_classification/01_Machine_Learning_Review.md`: Review of machine learning fundamentals, covering supervised, unsupervised, and reinforcement learning approaches.
*   `week_3_classification/02_Supervised_Machine_Learning_Concepts.md`: Detailed explanation of supervised learning concepts relevant to text classification.
*   `week_3_classification/03_Training_and_Evaluation.md`: Guidelines and best practices for training and evaluating text classification models.
*   `week_3_classification/04_Sentiment_Analysis.md`: In-depth exploration of sentiment analysis, including its applications and classifier architecture.
*   `week_3_classification/05_Implementation.md`: Step-by-step implementation of a sentiment classifier using Bag-of-Words and machine learning algorithms.

## Week 4: Dimensionality Reduction and Advanced Text Features

The fourth week explores techniques to optimize text classification by reducing dimensionality and enriching the Bag-of-Words model with contextual information.

### Topics Covered:

*   **Week Review:** A comprehensive review of the classification pipeline and introduction to optimization strategies.
*   **Stopword Removal:** Using stopword removal as a dimensionality reduction technique.
*   **Lemmatization:** Applying lemmatization for semantic grouping and significant dimensionality reduction.
*   **N-Grams:** Exploring N-grams to capture contextual information in text classification.
*   **Stemming:** Analyzing stemming as a faster alternative to lemmatization and comparing these techniques.

### Files:

*   `week_4_dimensionality_reduction/01_Week_Review.md`: Review of the classification pipeline with Bag-of-Words and introduction to optimization strategies.
*   `week_4_dimensionality_reduction/02_Stopword_Removal.md`: Detailed explanation of stopword removal using Scikit-learn and spaCy.
*   `week_4_dimensionality_reduction/03_Lemmatization.md`: In-depth coverage of lemmatization for semantic grouping and its implementation with spaCy.
*   `week_4_dimensionality_reduction/04_N-Grams.md`: Exploration of N-grams to capture contextual information and enhance classification performance.
*   `week_4_dimensionality_reduction/05_Stemming.md`: Analysis of stemming as a faster alternative to lemmatization with usage guidelines.

## Week 5: Topic Modeling and Document Clustering

The fifth week focuses on unsupervised learning techniques for text data, exploring document clustering and topic modeling approaches that discover thematic structures within text collections.

### Topics Covered:

*   **Bag-of-Words and Unsupervised Learning:** Understanding how the BoW representation can be utilized in unsupervised learning contexts.
*   **K-Means for Document Clustering:** Exploring how K-Means can group similar documents based on their vector representations.
*   **Topic Modeling with LDA:** Introduction to Latent Dirichlet Allocation, a probabilistic model that discovers abstract topics within document collections.

### Files:

*   `week_5_topic_modeling/01_BOW_and_Unsupervised_Learning.md`: Explanation of Bag-of-Words in the context of unsupervised learning applications.
*   `week_5_topic_modeling/02_K-Means_for_Document_Clustering.md`: Detailed guide on implementing K-Means clustering for document grouping.
*   `week_5_topic_modeling/03_Topic_Modeling_with_LDA.md`: In-depth exploration of LDA for topic discovery and document representation as topic mixtures.

## Week 6: Word Embeddings and Modern Text Representation

The sixth week explores word embeddings, a modern approach to text representation that captures semantic relationships between words and provides dense vector representations that significantly outperform traditional sparse methods.

### Topics Covered:

*   **Introduction to Embeddings:** Understanding the concept of word embeddings, their advantages over traditional methods like One-Hot Encoding, and how they capture meaningful semantic relationships.
*   **Practical Implementation with Gensim:** Learning how to use pre-trained word embeddings with the Gensim library, exploring word similarity, analogies, and sentence comparison techniques.
*   **Embeddings Training and Architecture:** Exploring how embeddings are trained using the Word2Vec algorithm, including Skip-gram and Continuous Bag-of-Words (CBOW) architectures.

### Files:

*   `week_6_embeddings/01_Embeddings_Introduction.md`: Thorough introduction to word embeddings, explaining the concept and how they capture semantic relationships between words.
*   `week_6_embeddings/02_Embeddings_Implementation.md`: Detailed guide on implementing and using pre-trained embeddings with the Gensim library.
*   `week_6_embeddings/03_Embeddings_Training.md`: In-depth exploration of Word2Vec training methodology, including Skip-gram and CBOW architectures.
