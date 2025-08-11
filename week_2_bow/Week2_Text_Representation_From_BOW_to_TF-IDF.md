# Week 2: Text Representation - From Bag-of-Words to TF-IDF

This week, we will study the Bag-of-Words (BOW) model, a fundamental technique for representing text documents in a way that computers can understand. We will explore its theoretical basis, its practical implementation, and then move on to a more advanced and widely used metric, TF-IDF.

## 1. The Bag-of-Words (BOW) Model

The core analogy of the BOW model is to treat a text document as an unordered "bag" of its words. The model disregards grammar and word order but keeps track of word frequency (multiplicity). Its primary goal is to convert text documents into numerical feature vectors, which can then be used in machine learning algorithms.

### Core Concepts

-   **Document:** A single piece of text. This can be a sentence, a paragraph, or an entire article.
-   **Corpus:** A collection of all documents used in a study.
-   **Vocabulary (or Dictionary):** A set of all unique words that appear across the entire corpus.

### How It Works

Let's illustrate with a simple corpus of three documents:

-   Document 1: "O gato pulou" (The cat jumped)
-   Document 2: "O gato caiu" (The cat fell)
-   Document 3: "O gato mia" (The cat meows)

**Step 1: Build the Vocabulary**
First, we create a vocabulary containing every unique word from our corpus:
`{O, gato, pulou, caiu, mia}`

**Step 2: Vectorize the Documents**
Next, we create a feature vector for each document. The length of each vector is equal to the size of our vocabulary. For each word in the vocabulary, we mark its presence (1) or absence (0) in the document.

-   **Document 1 ("O gato pulou"):** `[1, 1, 1, 0, 0]`
-   **Document 2 ("O gato caiu"):** `[1, 1, 0, 1, 0]`
-   **Document 3 ("O gato mia"):** `[1, 1, 0, 0, 1]`

Notice that the first two elements of every vector are `1`, correctly capturing that all sentences start with "O gato". The final elements differ, capturing the unique verb in each sentence.

### Training and Application

-   **Fit (Training/Adjustment):** The process of building the vocabulary from a given corpus is called "fitting" or "training" the model.
-   **Transform (Application):** Once the model is fitted, we can use its vocabulary to "transform" new, unseen documents into vectors.

For example, let's transform a new document: `"O cachorro pulou e caiu"` (The dog jumped and fell).

Using our existing vocabulary, the vector would be: `[1, 0, 1, 1, 0]`.

-   The model correctly identifies the presence of "O", "pulou", and "caiu".
-   The new words "cachorro" and "e" are not in our original vocabulary, so they are ignored.

This vector representation allows us to use mathematical measures like cosine similarity or Euclidean distance to compare documents and find similarities.

## 2. Practical Implementation with `CountVectorizer`

The Scikit-Learn library provides a simple way to implement the BOW model using the `CountVectorizer` class.

### Adjusting (Fitting) the Model

First, we fit the vectorizer on a corpus to build the vocabulary.

```python
from sklearn.feature_extraction.text import CountVectorizer

# 1. Define the corpus
corpus = [
    'Top Gear é um jogo de ação',
    'Mineração de dados é interessante'
]

# 2. Instantiate and fit the vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(corpus)

# 3. View the vocabulary
# The output maps each word to a column index
print(vectorizer.vocabulary_)
# {'top': 5, 'gear': 2, 'um': 6, 'jogo': 3, 'de': 1, 'ação': 0, 
#  'mineração': 4, 'dados': 0, 'é': 7, 'interessante': 8} 
```

**Key Defaults of `CountVectorizer`:**
-   Words are converted to lowercase.
-   By default, words (tokens) with fewer than two characters are ignored (e.g., the word "é" is dropped).
-   Punctuation is generally removed.

### Applying (Transforming) the Model

We use the `transform` method to create vector representations for new documents.

```python
# New documents to transform
new_docs = ['Top Gun foi um filme interessante de ação']

# Transform the documents
X = vectorizer.transform(new_docs)

# The result 'X' is a sparse matrix. To view it:
print(X.toarray())
# [[1 1 0 0 0 1 1 0]] 
#  ação, de, gear, jogo, mineração, top, um
```
This output vector indicates that the words "ação", "de", "top", and "um" from our vocabulary were present in the new document.

### Binary Representation

`CountVectorizer` can also operate in a binary mode, where it only records the presence (1) or absence (0) of a term, ignoring its frequency within the document. This is useful when the number of times a word appears is not considered important.

```python
# Using binary=True
binary_vectorizer = CountVectorizer(binary=True)
# ... fitting and transforming will now produce only 0s and 1s.
```

## 3. A More Advanced Metric: TF-IDF

While counting words is useful, some words are more significant than others. Common words like "the" or "a" appear frequently but often carry less meaning than rarer, more specific terms. **TF-IDF (Term Frequency-Inverse Document Frequency)** is a statistical measure that evaluates how important a word is to a document in a collection or corpus.

It is composed of two parts:

### Term Frequency (TF)

**TF measures how frequently a term appears in a document.** It's the ratio of the number of times a term `t` appears in a document `d` to the total number of terms in that document.

**Formula:** `TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)`

The intuition is that the more a word appears in a document, the more important it is *to that specific document*.

### Inverse Document Frequency (IDF)

**IDF measures how important a term is across the entire corpus.** It penalizes common words and gives more weight to words that are rare.

**Formula:** `IDF(t, D) = log( (Total number of documents in corpus D) / (Number of documents containing term t) )`

-   If a term appears in many documents, the ratio inside the `log` approaches 1, and the IDF will be close to 0.
-   If a term is rare, the ratio will be large, resulting in a higher IDF score.

**Implementation Note:** In practice (e.g., in Scikit-Learn), the formula is often smoothed to prevent division by zero and to moderate the weights of very rare terms. A common variant is:
`IDF(t, D) = log( (1 + N) / (1 + df(t)) ) + 1`
where `N` is the total number of documents and `df(t)` is the number of documents containing term `t`.

### The TF-IDF Score

The final TF-IDF score for a word is simply the product of its TF and IDF scores.

**Formula:** `TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)`

This gives a weight that is high when a term appears often in a specific document but rarely in the overall corpus, indicating high relevance for that document.

## 4. Practical Implementation with `TfidfVectorizer`

Scikit-Learn's `TfidfVectorizer` implements the TF-IDF logic, sharing a similar interface with `CountVectorizer`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Define the corpus
corpus = [
    'aprendizado profundo é parecido com mineração',
    'perceptron é um tipo de rede neural',
    'redes neurais são um tópico interessante',
    'mineração de dados em nuvem'
]

# 2. Instantiate and fit-transform the vectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

# The result is a sparse matrix with TF-IDF scores
print(X_tfidf.toarray())
```

### L2 Normalization

By default, `TfidfVectorizer` applies **L2 normalization** to its output vectors. This ensures that all document vectors have a length of 1. This is a standard practice that improves the performance of many distance-based algorithms by ensuring that document length does not affect similarity calculations. Because of this, the raw TF-IDF scores may not be immediately obvious from the formulas alone.

### Inspecting IDF Values

You can inspect the calculated IDF values for each word in the vocabulary directly from the fitted vectorizer:

```python
# The idf_ attribute stores the IDF value for each vocabulary term
print(tfidf_vectorizer.idf_)
```
This is useful for understanding which words the model considers most significant across the corpus.

## 5. Applications in Information Retrieval and Recommendation

The vector representations created by BOW and TF-IDF are powerful tools for building applications like:

-   **Textual Information Retrieval (Search Engines):** Finding documents that are relevant to a user's keyword query.
-   **Recommender Systems:** Recommending documents (e.g., articles, products) to a user based on their similarity to items the user has shown interest in.

### Proof-of-Concept: A Simple Search System

We can implement a basic search system by treating a user's query as a new document. We transform the query into its TF-IDF vector and then find the documents in our corpus that are most similar to it.

Here is a conceptual example using the TF-IDF matrix we generated earlier:

```python
import pandas as pd

# Assume X_tfidf is our TF-IDF matrix and vectorizer is our fitted TfidfVectorizer
# For clarity, let's put it in a DataFrame
df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# 1. Define a search query
query = "perceptron redes neurais"

# 2. Get the query terms that exist in our vocabulary
query_terms = [term for term in query.split() if term in df.columns]

# 3. Filter and rank documents based on the query terms
# This simple logic sums the TF-IDF scores for the query terms in each document
# A real system would use cosine similarity.
results = df[query_terms].sum(axis=1).sort_values(ascending=False)

print(results)
```

This would rank the documents, showing that the documents containing "perceptron", "redes", and "neurais" with high TF-IDF scores are the most relevant to the search.

### Real-World Considerations

-   **Scalability:** For large-scale systems with millions of documents (like web search), this process must be highly optimized. This involves creating an "index" of the TF-IDF data.
-   **Specialized Databases:** Modern databases like MongoDB and search platforms like Elasticsearch have built-in, highly efficient mechanisms for performing keyword-based text searches, often using these same underlying principles.
