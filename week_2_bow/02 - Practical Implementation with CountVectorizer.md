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