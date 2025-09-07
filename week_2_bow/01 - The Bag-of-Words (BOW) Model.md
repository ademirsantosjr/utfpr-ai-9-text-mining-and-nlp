## 1. The Bag-of-Words (BOW) Model

The core analogy of the BOW model is to treat a text document as an unordered "bag" of its words. The model disregards grammar and word order but keeps track of word frequency (multiplicity). Its primary goal is to convert text documents into numerical feature vectors, which can then be used in machine learning algorithms.

### Core Concepts

-   **Document:** A single piece of text. This can be a sentence, a paragraph, or an entire article.
-   **Corpus:** A collection of all documents used in a study.
-   **Vocabulary (or Dictionary):** A set of all unique words that appear across the entire corpus.

### How It Works

Let's illustrate with a simple corpus of three documents:

-   Document 1: "O gato pulou" (The cat jumped)
-   Document 2: "O gato caiu" (The cat fell)
-   Document 3: "O gato mia" (The cat meows)

**Step 1: Build the Vocabulary**
First, we create a vocabulary containing every unique word from our corpus:
`{O, gato, pulou, caiu, mia}`

**Step 2: Vectorize the Documents**
Next, we create a feature vector for each document. The length of each vector is equal to the size of our vocabulary. For each word in the vocabulary, we mark its presence (1) or absence (0) in the document.

-   **Document 1 ("O gato pulou"):** `[1, 1, 1, 0, 0]`
-   **Document 2 ("O gato caiu"):** `[1, 1, 0, 1, 0]`
-   **Document 3 ("O gato mia"):** `[1, 1, 0, 0, 1]`

Notice that the first two elements of every vector are `1`, correctly capturing that all sentences start with "O gato". The final elements differ, capturing the unique verb in each sentence.

### Training and Application

-   **Fit (Training/Adjustment):** The process of building the vocabulary from a given corpus is called "fitting" or "training" the model.
-   **Transform (Application):** Once the model is fitted, we can use its vocabulary to "transform" new, unseen documents into vectors.

For example, let's transform a new document: `"O cachorro pulou e caiu"` (The dog jumped and fell).

Using our existing vocabulary, the vector would be: `[1, 0, 1, 1, 0]`.

-   The model correctly identifies the presence of "O", "pulou", and "caiu".
-   The new words "cachorro" and "e" are not in our original vocabulary, so they are ignored.

This vector representation allows us to use mathematical measures like cosine similarity or Euclidean distance to compare documents and find similarities.