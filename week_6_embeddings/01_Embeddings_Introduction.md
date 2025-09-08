
# Embeddings: An Introduction to Word Representation

In the universe of Natural Language Processing (NLP), the way we represent words so that computers can understand them is fundamental. This document explores the concept of *Embeddings*, a powerful technique that has revolutionized the way we handle text in Machine Learning models.

## The Challenge of Word Representation

Machine learning models do not understand text in its raw format. They require words to be converted into numerical vectors. A traditional approach to this is **One-Hot Encoding**.

### The Limitations of One-Hot Encoding

In One-Hot Encoding, each word in a vocabulary is represented by a long vector, composed mostly of zeros and a single "1" in the position that corresponds to the word.

For example, in a vocabulary of ["king", "queen", "man", "woman"]:
- `king` could be `[1, 0, 0, 0]`
- `queen` would be `[0, 1, 0, 0]`

This approach has two major disadvantages:
1.  **High Dimensionality**: For a large vocabulary (e.g., 50,000 words), each vector would have 50,000 dimensions, making processing computationally expensive.
2.  **Lack of Semantic Relationship**: The resulting vectors are orthogonal to each other. This means that the model cannot infer similarity relationships. Mathematically, the distance between "king" and "queen" is the same as the distance between "king" and "banana", which does not reflect the reality of language.

## Word Embeddings: The Dense and Semantic Solution

To overcome the limitations of One-Hot Encoding, **Word Embeddings** emerged. The central idea is to represent words in a dense vector space of much smaller dimension (usually between 50 and 300 dimensions).

In this space, the position of each word vector is learned from the context in which it appears. As a result, words with similar meanings are positioned close to each other.

### Capturing Semantic Relationships

The great advantage of embeddings is their ability to capture the meaning and semantic relationships between words.

-   **Similarity**: Words like "cat" and "dog" will have vectors closer to each other than "cat" and "car".
-   **Analogies**: The vectors preserve analogical relationships. The famous relationship `vector("king") - vector("man") + vector("woman")` results in a vector very close to `vector("queen")`. This demonstrates that the model has learned the concept of "gender" and "royalty" implicitly.

## Visualizing Embeddings

To make the concept more tangible, we can use dimensionality reduction techniques such as **t-SNE (t-Distributed Stochastic Neighbor Embedding)** to visualize the position of words in a 2D or 3D space. By doing so, it is possible to observe "clusters" of words with related meanings.

## Practical Example: Using Pre-trained Embeddings

Embeddings can be trained from scratch or, more commonly, we can use pre-trained models on vast text corpora. Below is an example of how to load and use pre-trained embeddings in Python with the `gensim` library.

```python
import gensim.downloader as api

# Loads a pre-trained embedding model (GloVe, trained on Twitter)
# The model has 25 dimensions
model = api.load("glove-twitter-25")

# Find the most similar words to "python"
print("Similar words to 'python':")
print(model.most_similar("python"))
# Expected output: [('programming', 0.90), ('code', 0.89), ...]

# Solve the analogy: king - man + woman = queen
print("
Solving the analogy 'king - man + woman':")
result = model.most_similar(positive=['king', 'woman'], negative=['man'])
print(result)
# Expected output: [('queen', 0.85), ...]
```

This simple example illustrates the power of embeddings to capture complex relationships of human language in a way that computational models can process efficiently.

---

Next, we will explore how these embeddings are implemented and trained in practice.
