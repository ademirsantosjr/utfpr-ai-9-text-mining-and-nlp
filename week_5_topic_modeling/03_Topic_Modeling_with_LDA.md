# Topic Modeling with Latent Dirichlet Allocation (LDA)

While algorithms like K-Means assign each document to a single cluster, documents often address multiple themes. **Topic Modeling** is an area of NLP that aims to discover the abstract thematic structures in a body of text. **Latent Dirichlet Allocation (LDA)** is the main algorithm for this task.

## What is LDA?

LDA is a generative statistical model that operates under two main premises:

1.  **Each document is a mixture of topics**: Instead of belonging to a single group, a document is seen as a combination of several topics in different proportions (e.g., 60% "electronics," 30% "logistics," 10% "customer service").
2.  **Each topic is a distribution of words**: A topic is defined by a set of words that frequently occur together. For example, the "electronics" topic would be characterized by words like "screen," "battery," "camera," "cell phone," etc.

LDA is a probabilistic model that, in an unsupervised way, analyzes the corpus and determines the topic structure and the composition of each document.

## Preparing Data for LDA

Unlike other models that can benefit from metrics like TF-IDF, LDA works based on word frequency. For this reason, the ideal input representation for LDA is a term count matrix. In `scikit-learn`, this is done using the `CountVectorizer`.

## Practical Implementation with Scikit-Learn

Let's continue using the same pre-processed and lemmatized corpus of negative reviews from the previous steps.

### 1. Feature Extraction with `CountVectorizer`

First, we create the term count matrix, which will serve as input for our LDA model.

```python
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en import stop_words

# Reusing the `corpus_preprocessed` from the previous steps
stop_words_en = list(stop_words.STOP_WORDS)
extractor_count = CountVectorizer(
    stop_words=stop_words_en,
    binary=True # Use binary counting (presence/absence) as in the notebook
)
X_lda = extractor_count.fit_transform(corpus_preprocessed)
```

### 2. LDA Model Training

With the count matrix, we instantiate and train the LDA model. The `n_components` hyperparameter defines the number of topics we want to discover.

```python
from sklearn.decomposition import LatentDirichletAllocation as LDA

# n_components defines the number of topics (default is 10)
model_lda = LDA(n_components=10, random_state=42)
model_lda.fit(X_lda)
```

## Interpreting the Topics

The most important result of LDA is the topics themselves. They are stored in `model_lda.components_`, where each topic is represented by a distribution of weights over all the words in the vocabulary. To make them interpretable, we can visualize the most important words of each topic.

```python
# Retrieve the vocabulary from our extractor
terms_dict = extractor_count.get_feature_names_out()

for id, topic in enumerate(model_lda.components_):
    # Get the indices of the 10 most important words and print them
    top_terms_indices = topic.argsort()[-10:]
    top_terms = [terms_dict[i] for i in top_terms_indices]
    print(f"Topic {id}: {', '.join(top_terms)}")

# Example Output:
# Topic 0: battery, stay, last, day, buy, other, product, not, like
# Topic 5: fragrance, other, seem, perfume, buy, device, fixation, price, product, like
# Topic 3: defect, stay, assistance, buy, work, problem, technical, use, product, like
```

Through the keywords, we can assign a human-readable label to each topic, such as "Battery Problems," "Perfumes and Cosmetics," and "Defects and Technical Assistance."

## Analyzing the Topic Distribution in Documents

We can also take a specific document and see its topic composition using the `transform` method.

```python
# Take an example document
doc_example = [corpus_raw[46]] # "It's good, but I wouldn't buy it again... Terrible fixation..."

# Preprocessing and vectorization
doc_preprocessed = preprocessing(doc_example)
doc_features = extractor_count.transform(doc_preprocessed)

doc_topic_distribution = model_lda.transform(doc_features)

# Show the probability of each topic for the document
print(doc_topic_distribution)

# Get the most likely topic
top_topic = doc_topic_distribution[0].argmax()
print(f"\nThe most likely topic for the document is Topic {top_topic}.")
```

By analyzing a document about the "terrible fixation" of a perfume, the LDA model correctly associates it with the topic that contains the words "fixation," "perfume," and "fragrance."

## Conclusion

LDA is a sophisticated technique that goes beyond simple clustering. It allows for a more granular analysis of the themes present in a collection of documents, being extremely useful for:

-   **Trend Analysis**: Discovering what people are talking about on social media or in product reviews.
-   **Document Organization**: Automatically classifying large volumes of articles, emails, or papers.
-   **Recommendation Systems**: Recommending content based on a user's topics of interest.

```