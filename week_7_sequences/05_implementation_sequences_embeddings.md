# Implementation: Vectorization, Embeddings and Classification

In this section, we will put theory into practice. We will use the Keras library to build a sentiment classification model that uses sequences and embeddings. The process is divided into clear steps, from text preparation to the construction and training of the neural network.

## Step 1: Generating Sequences with `TextVectorization`

The first step is to transform our raw text sentences into sequences of integers. The `TextVectorization` layer from Keras is perfect for this, as it takes care of creating the vocabulary, converting to indices, and padding in a single object.

### How it Works

1.  **Instantiation:** We create an instance of the layer. We can configure important hyperparameters such as `max_tokens` (the vocabulary size) and `output_sequence_length` (the fixed length of the sequences).
2.  **Adaptation (`.adapt()`):** The layer analyzes the training corpus to build the vocabulary, mapping the most frequent words to indices.
3.  **Vectorization:** Once adapted, the layer can be used to transform any text into sequences of integers.

### Code Example

```python
from keras.layers import TextVectorization
import numpy as np

corpus = ["The dog barks and jumps.",
          "The cat meows and jumps.",
          "The dog does not meow."]

# 1. Instantiates and adapts the layer to the corpus
vectorizer_layer = TextVectorization()
vectorizer_layer.adapt(corpus)

# 3. Generates the sequences
sequences = vectorizer_layer(corpus)

print("Generated sequences:")
print(sequences)
print("\nMapped vocabulary:")
print(vectorizer_layer.get_vocabulary())
```

## Step 2: The `Embedding` Layer

With the sequences ready, the next step is to use the `Embedding` layer to convert the indices into dense vectors.

- **`input_dim`**: The size of the vocabulary + 1 (the `+1` is for the padding/unknown token).
- **`output_dim`**: The number of dimensions we want for our embedding vectors (e.g., 50, 100, 300).

When added to a model, the `Embedding` layer transforms an input of shape `(batch_size, sequence_length)` into an output of shape `(batch_size, sequence_length, embedding_dim)`.

## Step 3: Classification Model with Embeddings

Let's now build a classification model for the Buscapé sentiment analysis dataset. The architecture will be:

1.  `TextVectorization`
2.  `Embedding`
3.  `Flatten` or `Reshape` (to flatten the 3D output of the embedding to 2D, compatible with the `Dense` layer)
4.  `Dense` (hidden layer with ReLU activation)
5.  `Dense` (output layer with Sigmoid activation for binary classification)

### Code Example

```python
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

# Hyperparameters
VOCAB_SIZE = 20000
MAX_SEQUENCE_SIZE = 150
EMBEDDING_DIM = 50
NEURONS = 30

# (Assuming X_train and y_train have already been loaded)

# Vectorization layer
vectorization_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=MAX_SEQUENCE_SIZE
)
vectorization_layer.adapt(X_train)

# Model Construction
model = Sequential([
    vectorization_layer,
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    Flatten(), # Flattens the output from (None, 150, 50) to (None, 150 * 50)
    Dense(NEURONS, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Training
# model.fit(X_train, y_train, epochs=5, validation_split=0.1)
```

Using the `Embedding` layer usually leads to a significant increase in accuracy compared to models that do not use it, as the model can learn semantic representations during training.

## Step 4: Transfer Learning with Pre-trained Embeddings

We can further improve performance and accelerate convergence by using **pre-trained embeddings**. The idea is to initialize the `Embedding` layer with weights from vectors that have already been trained on a massive corpus (such as Word2Vec trained on all of Wikipedia's content).

The process is:

1.  **Load the Vectors:** We use a library like `Gensim` to load a pre-trained embeddings file (e.g., `skip_s50.txt`).
2.  **Build the Weight Matrix:** We create a matrix where row `i` contains the vector of the `i`-th token of our vocabulary (`vectorization_layer.get_vocabulary()`).
3.  **Initialize the `Embedding` Layer:** We pass this weight matrix to the `weights` parameter of the `Embedding` layer and, optionally, set `trainable=False` to freeze the weights and only train the rest of the network.

### Code Example (Matrix construction logic)

```python
# (Assuming 'vectors' has been loaded with Gensim and 'vectorization_layer' has been adapted)
vocabulary = vectorization_layer.get_vocabulary()
word_index = dict(zip(vocabulary, range(len(vocabulary))))

# Create the weight matrix
embedding_matrix = np.zeros((len(vocabulary), EMBEDDING_DIM))
for word, i in word_index.items():
    if word in vectors:
        embedding_matrix[i] = vectors[word]

# Use the matrix to initialize the Embedding layer
embedding_layer = Embedding(
    input_dim=len(vocabulary),
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix], # Pass the weight matrix
    trainable=False # Optional: freeze the weights
)
```

This *Transfer Learning* technique injects a vast amount of linguistic knowledge into the model from the beginning, which is especially useful when our training dataset is small.

