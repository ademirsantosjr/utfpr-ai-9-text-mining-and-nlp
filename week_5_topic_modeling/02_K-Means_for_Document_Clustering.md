# K-Means for Text Document Clustering

After converting texts into numerical vectors using techniques like Bag-of-Words or TF-IDF, we can apply unsupervised learning algorithms to discover hidden patterns in the data. **K-Means** is one of the most popular and effective clustering algorithms for grouping documents by thematic similarity.

## The K-Means Algorithm

K-Means is an iterative algorithm that aims to partition a dataset into **K** distinct clusters, where each document belongs to the cluster with the nearest centroid (the center of the cluster).

The operational logic can be summarized in the following steps:

1.  **Initialization**: **K** centroids are randomly defined in the vector space. Each centroid is the initial center point of a cluster.
2.  **Assignment**: Each document (vector) in the corpus is assigned to the cluster whose centroid is closest. Proximity is usually measured by Euclidean distance.
3.  **Update**: After assigning all documents, the centroid of each cluster is recalculated to be the mean of all documents belonging to that cluster.

Steps 2 and 3 are repeated until the position of the centroids does not change significantly between iterations or until a maximum number of iterations is reached. The main hyperparameter of the model is the number of clusters, **K**, which must be specified before training.

## Applying K-Means to Text

The workflow for applying K-Means to a text corpus is as follows:

`Text Corpus → Preprocessing → Vectorization (TF-IDF) → K-Means Training`

Once the documents are represented as vectors, K-Means groups them based on their proximity in the vector space. Documents that share a similar vocabulary (and, therefore, have close vectors) will be grouped in the same cluster, revealing thematic groups.

## Practical Implementation with Scikit-Learn

Let's demonstrate the application of K-Means on a dataset of product reviews, focusing on comments with negative sentiment.

### 1. Data Loading and Preprocessing

First, we load the data, filter the negative reviews, and perform preprocessing, which includes **lemmatization** to reduce words to their base forms.

```python
import pandas as pd
import spacy
from spacy.lang.en import stop_words

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Load and filter the dataset
df = pd.read_csv('./buscape.csv')
df_filtered = df.loc[df['polarity']==0].dropna()
corpus_raw = df_filtered['review_text'].tolist()

# Preprocessing function with lemmatization
def preprocessing(corpus):
    corpus_with_lemmas = []
    for text in corpus:
        doc = nlp(text.lower())
        corpus_with_lemmas.append(' '.join([token.lemma_ for token in doc]))
    return corpus_with_lemmas

corpus_preprocessed = preprocessing(corpus_raw)
```

### 2. Feature Extraction with TF-IDF

With the preprocessed text, we use the `TfidfVectorizer` to convert the corpus into a matrix of TF-IDF vectors, also removing stop words.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words_en = list(stop_words.STOP_WORDS)
extractor = TfidfVectorizer(stop_words=stop_words_en)
X = extractor.fit_transform(corpus_preprocessed)
```

### 3. K-Means Model Training

Now, with the `X` matrix ready, we can train the K-Means model.

```python
from sklearn.cluster import KMeans

# Instantiate and train the model
# n_init='auto' allows scikit-learn to determine the number of initializations
model = KMeans(n_init='auto', random_state=42)
model.fit(X)
```

## Analysis and Exploration of Clusters

After training, the `model.labels_` attribute contains the cluster ID for each document in the corpus. We can use this information to analyze the formed groups.

```python
# Create a DataFrame to facilitate analysis
clusters_df = pd.DataFrame()
clusters_df['text'] = corpus_raw
clusters_df['cluster'] = model.labels_

# Example: Analyzing a document and its cluster neighbors
doc_index = 77
doc_text = clusters_df.iloc[doc_index]['text']
doc_cluster_id = clusters_df.iloc[doc_index]['cluster']

print(f"Original Text: {doc_text}")
print(f"Cluster ID: {doc_cluster_id}")

# Show 5 other documents from the same cluster
similar_docs = clusters_df[clusters_df['cluster'] == doc_cluster_id].head()
print("\nDocuments in the same cluster:")
print(similar_docs)
```

By running the code above, it is observed that the example document, which complains about "poor sound quality," is grouped with other documents that also mention "sound" problems, validating the ability of K-Means to group by theme.

## Classifying New Documents

The trained model can also be used to determine which cluster a new document would belong to.

```python
# New document
new_doc = "Don't buy this cell phone. The battery doesn't last at all."

# Preprocessing and vectorization (using the same trained extractor)
new_doc_preprocessed = preprocessing([new_doc])
new_doc_features = extractor.transform(new_doc_preprocessed)

# Cluster prediction
[cluster_id] = model.predict(new_doc_features)
print(f"The new document belongs to cluster: {cluster_id}")
```

## Conclusion

K-Means is a powerful tool for exploratory analysis of textual data. It allows for the automatic identification of groups of documents with similar semantic characteristics in large volumes of text, and is widely used in applications such as:

-   Customer feedback analysis.
-   Organization of articles and news.
-   Content recommendation systems.

