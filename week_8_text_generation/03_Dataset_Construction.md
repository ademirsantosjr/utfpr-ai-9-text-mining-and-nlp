
# Building the Dataset for Text Generation

With the corpus cleaned and prepared, the next step is to transform it into a format that can be used to train a machine learning model. Text generation, in this context, is treated as a **supervised learning** problem.

-   **Input (X):** A sequence of words (context).
-   **Output (y):** The next word in the sequence (label).

Our goal is to build a dataset with multiple pairs of `(context, label)` from our corpus.

## 1. Tokenization with spaCy

First, we need to break our documents (which are strings) into smaller units, the **tokens** (words). For this task, we use the `spaCy` library, a powerful framework for natural language processing. As we only need the tokenization functionality, we disable other more complex components of the `spaCy` pipeline to optimize performance.

```python
import spacy

# Loads the Portuguese model, disabling unnecessary components
pln = spacy.load("pt_core_news_sm", disable=[
    "morphologizer", "senter", "attribute_ruler", "ner"])
```

## 2. The Sliding Window Algorithm

To create our `(X, y)` pairs, we use the **sliding window** technique. The idea is to go through the text with a "window" of a fixed size. The content inside the window becomes our input `X`, and the immediately following word becomes our label `y`.

For example, consider the sentence "deep neural networks are a technique" and a **window of size 4**:

1.  **Window 1:**
    -   `X`: "deep neural networks are"
    -   `y`: "a"
2.  **Window 2 (slides one word to the right):**
    -   `X`: "neural networks are a"
    -   `y`: "technique"

This process is repeated over the entire corpus, generating a large volume of training samples. The **window size (`window_size`)** is an important hyperparameter: larger windows provide more context to the model, but increase computational complexity.

The code below implements the sliding window strategy to build the dataset.

```python
import numpy as np

window_size = 30

X = []
labels = []

for text in corpus:
  doc = list(pln(text))
  tokens = [ token.text for token in doc ]

  # Iterates over the tokens to create the windows
  for i in range(0, len(tokens)-1):
    context = tokens[max(i-window_size, 0):i]
    label = tokens[i]

    X.append(' '.join(context))
    labels.append(label)

X = np.array(X, dtype="object")
```

## 3. Preparing the Labels for Keras

The output layer of our model (`Dense` with `Softmax` activation) requires that the output labels be numerical and in a specific format, known as **One-Hot Encoding**. We cannot simply pass the words as strings.

O processo de conversão consiste em:

1.  **Create an Output Vocabulary:** Generate a list of all the unique words that appear as a label (`labels`). This will be our `word_index`.
2.  **Map Labels to Indices:** Replace each word in `labels` with its corresponding index in the `word_index`.
3.  **Apply One-Hot Encoding:** Use the `to_categorical` function from Keras to convert the vector of indices into a binary matrix. Each row of this matrix will have the length of the vocabulary, with the value `1` in the position of the word's index and `0` in the others.

```python
from keras.utils import to_categorical

# 1. Create the output vocabulary
word_index = list(set(labels))

# 2. Map labels to indices
labels_index = [word_index.index(label) for label in labels]

# 3. Apply One-Hot Encoding
y = to_categorical(labels_index)
```

The dimensionality of the resulting `y` matrix will be `(number_of_samples, vocabulary_size)`. The `vocabulary_size` (in this case, `len(word_index)`) determines the number of neurons that our `Softmax` output layer will need to have.
