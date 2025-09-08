# Word Embeddings: Adding Semantic Meaning to Sequences

After transforming our documents into sequences of numerical indices, the next step is to enrich this representation with **semantic meaning**. This is where **Word Embeddings** come in.

While sequences preserve the order of words, embeddings help us capture the *meaning* behind them.

## What are Embeddings?

An *embedding* is a dense vector representation of a word or token. Instead of a simple integer, each word in the vocabulary is mapped to a vector of floating-point values (for example, a vector of 100, 200, or 300 dimensions).

The goal is for these vectors to represent the semantics of the tokens in an n-dimensional space. In this space, words with similar meanings (like "king" and "queen") or contextual relationships (like "France" and "Paris") will be positioned close to each other.

## From Index to Vector: The Embedding Layer

In a Deep Learning model, the transformation of the sequence of indices into a sequence of vectors is performed by an **Embedding layer**.

The process works as follows:

1.  **Input:** The layer receives the sequence of integer indices that we created earlier (with padding).
2.  **Lookup Table:** The embedding layer maintains an internal table where each token index is associated with an embedding vector.
3.  **Output:** For each index in the input sequence, the layer replaces it with the corresponding embedding vector. The output is a matrix (or a sequence of vectors) that will serve as input for the next layers of the neural network.

### Illustration of the Process

The transformation flow can be visualized as follows:

```
# 1. Sequence of Indices (Input)
[2, 3, 5, 6, 0, 0]
   |
   v
+--------------------+
| Embedding Layer    |
+--------------------+
   |
   v
# 2. Sequence of Vectors (Output)
[
  [0.12, 0.45, ..., 0.81],  # Vector for index 2 ("john")
  [0.67, 0.33, ..., 0.19],  # Vector for index 3 ("won")
  [0.91, 0.05, ..., 0.42],  # Vector for index 5 ("race")
  [0.28, 0.76, ..., 0.55],  # Vector for index 6 (".")
  [0.00, 0.00, ..., 0.00],  # Vector for padding (0)
  [0.00, 0.00, ..., 0.00]   # Vector for padding (0)
]
```

## The Power of Semantic Representation

The great advantage of using embeddings is that they allow the model to generalize its learning. If the model learns something about the word "dog", it can apply similar knowledge to the word "puppy", as their embedding vectors will be very close.

By replacing a term with a synonym, the vector representation of the sequence changes very little, which makes the model more robust and accurate in understanding natural language.
