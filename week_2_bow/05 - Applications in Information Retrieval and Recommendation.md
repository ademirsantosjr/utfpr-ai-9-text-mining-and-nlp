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