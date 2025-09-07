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