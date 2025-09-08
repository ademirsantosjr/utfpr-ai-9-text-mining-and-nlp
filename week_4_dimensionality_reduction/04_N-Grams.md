
# Strategy 3: Capturing Context with N-Grams

So far, our strategies have focused on reducing dimensionality. Now, let's explore a technique that, although it increases dimensionality, drastically enriches the meaning captured by the model: **N-grams**.

## The Limitation of Bag-of-Words: The Lack of Order

The Bag-of-Words (BoW) model ignores the order of words. For it, the phrases "do not recommend the product" and "recommend the not product" may seem very similar if the vocabulary is the same. The negation "not", which reverses the meaning of the sentence, loses its contextual strength when analyzed in isolation.

This lack of context is one of the biggest weaknesses of BoW.

## What are N-Grams?

N-grams are contiguous sequences of *n* words in a text. Instead of looking only at isolated words (**unigrams**), we can extract features from pairs of words (**bigrams**), triplets (**trigrams**), and so on.

-   **Text:** "the cat jumped high"
-   **Unigrams (1-gram):** `["the", "cat", "jumped", "high"]`
-   **Bigrams (2-grams):** `["the cat", "cat jumped", "jumped high"]`
-   **Trigrams (3-grams):** `["the cat jumped", "cat jumped high"]`

By using bigrams like "do not recommend", we capture the negative sentiment much more effectively than with the isolated words "not" and "recommend".

## Implementation with Scikit-learn

Scikit-learn's vectorizers have the `ngram_range` parameter, which defines the size range of the N-grams to be extracted. It is a tuple `(min_n, max_n)`.

```python
from sklearn.feature_extraction.text import CountVectorizer

# To extract unigrams, bigrams, and trigrams
extractor_with_ngrams = CountVectorizer(
    ngram_range=(1, 3), # (min_n, max_n)
    stop_words=stopwords_pt # We still use the previous optimizations
)

# The input corpus should still be the lemmatized one
# for maximum efficiency.
```

## Analysis of Results: The Trade-off between Context and Dimensionality

### 1. The Explosion of Features

The main consequence of using N-grams is a massive increase in the number of features, as each sequence of words becomes a new column in the matrix.

| Strategy Applied | No. of Features |
| :--- | :---: |
| Stopwords + Lemmatization | 42,877 |
| **+ N-grams (1, 3)** | **1,616,716** |

We went from ~42 thousand to **over 1.6 million features**. This is an increase that requires more memory and computational power.

### 2. The Gain in Performance

The computational cost was worth it. The context added by the N-grams allowed the model to make much more subtle distinctions, improving performance across all metrics, especially for the minority class.

| Metric | Without N-grams | With N-grams (1, 3) | Variation |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 0.91 | **0.93** | +0.02 |
| **F1-Score (Negative)** | 0.48 | **0.58** | **+0.10** |
| **Weighted F1-Score** | 0.90 | **0.92** | +0.02 |

The **10-point increase in the F1-Score of the negative class** is the most expressive result so far, showing that the model has finally begun to better understand the negative comments.

## Roadmap for Advanced Optimization

The techniques we have seen form the basis of text classification. To extract even more performance, the next step is systematic experimentation and optimization. Here is a roadmap:

1.  **Test Vectorizers**: Replace `CountVectorizer` with `TfidfVectorizer`, which weights the importance of terms.
2.  **Explore Classifiers**: Replace the Decision Tree with more robust algorithms like `LogisticRegression`, `LinearSVC`, or `RandomForest`.
3.  **Feature Selection**: To deal with the high dimensionality of N-grams, use feature selection techniques like `SelectKBest` to keep only the most informative features.
4.  **Hyperparameter Optimization**: Use `GridSearchCV` to find the best combination of parameters for the vectorizer, feature selector, and classifier.
5.  **Build a Pipeline**: Join all the steps (`vectorizer` -> `selector` -> `classifier`) into a Scikit-learn `Pipeline`. This organizes the code and ensures that the data is processed correctly during cross-validation.

By following a roadmap like this, it is possible to achieve even better results. In a more complete example, an optimized configuration with `RandomForest` and `TfidfVectorizer` achieved an **F1-Score of 0.62** for the minority class.
