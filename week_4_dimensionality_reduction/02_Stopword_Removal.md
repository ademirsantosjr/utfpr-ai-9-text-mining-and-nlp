
# Strategy 1: Stopword Removal for Dimensionality Reduction

One of the most common and effective techniques for dimensionality reduction in Natural Language Processing is the removal of *stopwords*.

## What are Stopwords?

Stopwords are high-frequency words in a language that carry little or no semantic meaning for text analysis. Generally, they are articles, prepositions, conjunctions, and pronouns (e.g., "the", "a", "of", "to", "with", "that").

In the context of the Bag-of-Words model, these words appear in almost all documents, becoming features that do not help distinguish the content or sentiment of a text. By removing them, we reduce noise and the number of dimensions (terms) in our dictionary, which can lead to simpler and faster models.

## Implementation with Scikit-learn and spaCy

The most practical approach is to use a predefined list of stopwords and instruct our vectorizer to ignore them during the creation of the dictionary and the vectorization of texts.

### Step 1: Get the List of Stopwords

The `spaCy` library offers curated lists of stopwords for various languages, including Portuguese.

```python
# It is necessary to have spaCy and the Portuguese model installed
# pip install spacy
# python -m spacy download pt_core_news_sm

import spacy
from spacy.lang.pt.stop_words import STOP_WORDS

# STOP_WORDS is a set of words
print(f"Number of stopwords in Portuguese: {len(STOP_WORDS)}")
# print(list(STOP_WORDS)[:10]) # Displays the first 10
```

### Step 2: Configure the Vectorizer

Scikit-learn's vectorizers (`CountVectorizer` and `TfidfVectorizer`) accept a `stop_words` parameter in their constructor. Simply pass the list we obtained.

```python
from sklearn.feature_extraction.text import CountVectorizer

# List of stopwords from spaCy
stopwords_pt = list(STOP_WORDS)

# Instantiates the vectorizer configured to remove stopwords
extractor_with_stopwords = CountVectorizer(stop_words=stopwords_pt)

# This extractor will now ignore stopwords during the fit_transform process
```

## Analysis of Results

By applying this strategy in our sentiment classification pipeline, we observe two main effects:

### 1. Dimensionality Reduction

The number of unique terms in the dictionary (and, consequently, the number of columns in the Bag-of-Words matrix) decreased.

- **Without stopword removal:** 49,551 features
- **With stopword removal:** 49,163 features

This confirms that the stopwords were successfully removed, resulting in a leaner vector representation.

### 2. Model Performance

We evaluated the model trained with and without stopword removal, using the `train_and_validate` function.

| Metric | Without Stopwords | With Stopwords | Variation |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 0.91 | 0.91 | None |
| **Precision (Negative)** | 0.50 | 0.48 | -0.02 |
| **Recall (Negative)** | 0.45 | 0.47 | +0.02 |
| **F1-Score (Negative)** | 0.47 | 0.47 | None |

**Observation:** Although the overall accuracy remained the same, there were slight fluctuations in the precision and recall of the minority class (negative). However, the F1-Score, which is the harmonic mean of both, remained stable. This indicates that the removed words were not essential for the classification task.

## Conclusion

Stopword removal proved to be a valid strategy. **We reduced the complexity of the data** to be processed by the classifier, which can decrease training time, **without sacrificing the overall performance of the model**.
