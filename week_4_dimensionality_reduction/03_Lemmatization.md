
# Strategy 2: Lemmatization for Semantic Grouping

Continuing the search for a more efficient model, the next dimensionality reduction strategy is **lemmatization**.

## What is Lemmatization?

Lemmatization is the process of reducing a word to its base or dictionary form, known as the **lemma**. Unlike other techniques, lemmatization takes into account the morphological context of the word to make the conversion.

For example, in the standard Bag-of-Words model, words like "was", "go", "would go", and "let's go" would be treated as four distinct features. Lemmatization is smart enough to understand that they all derive from the verb **"to go"** and groups them under this lemma.

**Examples:**
- `has`, `had`, `having` → `to have`
- `game`, `playing`, `played` → `to play`
- `beautiful`, `lovely`, `pretty` → `beautiful`

By doing this, we consolidate features that share the same central meaning, promoting a much more aggressive and semantically coherent dimensionality reduction.

## Implementation with spaCy

Unlike stopword removal (which can be done directly in the vectorizer), lemmatization requires a **preprocessing step on the corpus** before vectorization.

The flow is as follows:
1.  For each document (text) in the corpus...
2.  ...process it with `spaCy` to get its tokens...
3.  ...and replace each token with its respective lemma (`token.lemma_`).

The result is a new corpus, where all words have been lemmatized.

```python
import spacy

# Loads the Portuguese model from spaCy
nlp = spacy.load("pt_core_news_sm")

def lemmatize_corpus(corpus):
    """Applies lemmatization to a list of documents."""
    lemmatized_corpus = []
    # Use nlp.pipe to process texts more efficiently
    for doc in nlp.pipe(corpus):
        # Reconstructs the document with the lemmas of each token
        lemmatized_doc = " ".join([token.lemma_ for token in doc])
        lemmatized_corpus.append(lemmatized_doc)
    return lemmatized_corpus

# Apply the function to the training and test data
# corpus_train_lemmatized = lemmatize_corpus(corpus_train)
# corpus_test_lemmatized = lemmatize_corpus(corpus_test)
```

## Analysis of Results

After preprocessing the corpus with lemmatization and then applying stopword removal in the vectorizer, the impact on dimensionality was remarkable.

### 1. Drastic Dimensionality Reduction

The combination of techniques resulted in a significant reduction in the number of features.

| Strategy Applied | No. of Features |
| :--- | :---: |
| Baseline (no optimizations) | 49,551 |
| + Stopword Removal | 49,163 |
| **+ Lemmatization** | **42,877** |

We reduced the dictionary by **more than 6,000 terms** compared to simple stopword removal, creating a significantly simpler model.

### 2. Model Performance

Even with a much smaller number of features to analyze, the classifier's performance remained stable.

| Metric | With Stopwords | With Stopwords + Lemmatization | Variation |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 0.91 | 0.91 | None |
| **Weighted F1-Score** | 0.90 | 0.90 | None |

This proves that the semantic information lost when grouping words into lemmas was not crucial for the sentiment classification task, validating the effectiveness of the technique.

## Computational Cost: A Point of Attention

Lemmatization is a computationally intensive task. Processing our dataset can take several minutes (in the class example, 10 to 15 minutes on Google Colab).

**Best Practice:** As it is a preprocessing step, it should be performed **only once**. Save the lemmatized corpus to a file so that it can be loaded quickly in future experiments, without the need for reprocessing.

## Conclusion

Lemmatization is a powerful tool for reducing dimensionality by grouping words based on their root meaning. Although it requires a greater computational investment in preprocessing, it drastically simplifies the final model while maintaining its performance intact.
