# Implementation: Creating a Base Model with Sequences

Before diving into more complex architectures, it is essential to establish a **baseline**. We will build and evaluate a first sentiment classification model that uses sequences, but **still without the Embedding layer**. This will allow us to clearly measure the impact that embeddings will have on the model's performance in the next step.

## Detailing the `TextVectorization` Layer

As introduced, the `TextVectorization` layer from Keras is our starting point. Let's revisit its main practical aspects:

1.  **Instantiation and Parameterization:**
    - `max_tokens`: Defines the maximum size of the vocabulary. Less frequent words will be treated as unknown (UNK).
    - `output_sequence_length`: Defines the fixed length of all output sequences. Longer sequences are **truncated** (*truncation* step), and shorter ones are filled with zeros (*padding* step).

2.  **Adaptation (`.adapt()`):**
    - This method is crucial. It processes the training corpus, builds the vocabulary with the `max_tokens` most common words, and prepares the layer for vectorization.

3.  **Use in the Model:**
    - Once adapted, the layer is added as the first layer of a Keras model, ensuring that any raw text passed to the model is automatically transformed into sequences of integers.

### Code Example: Configuring the Layer

```python
from keras.layers import TextVectorization

# Definition of hyperparameters
VOCAB_SIZE = 20000
MAX_SEQUENCE_SIZE = 150

# (Assuming X_train has already been loaded)

# 1. Instantiates the layer with the desired parameters
vectorization_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=MAX_SEQUENCE_SIZE
)

# 2. Adapts the layer to our training set
vectorization_layer.adapt(X_train)
```

## Building the Neural Network (Without Embeddings)

Our base model will be a simple (shallow) neural network that receives the sequences of integers and tries to learn to classify sentiments directly from them.

**Architecture:**
1.  `TextVectorization` (Input)
2.  `Dense` (Hidden layer with 30 neurons and `relu` activation)
3.  `Dense` (Output layer with 1 neuron and `sigmoid` activation)

### Code Example: Complete Model

```python
from keras.models import Sequential
from keras.layers import Dense

# (vectorization_layer already adapted from the previous cell)

# Construction of the Sequential Model
model = Sequential([
    vectorization_layer,
    Dense(30, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilation of the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Training the model for 5 epochs
# model.fit(X_train, y_train, epochs=5, validation_split=0.1)
```

## Analysis of the Base Model Results

When training and evaluating this model, we observe that the accuracy results are not very high. The model has particular difficulty in correctly identifying comments from the negative class (class 0).

**Conclusion:** The sequences, by themselves, do not provide enough semantic information for the model. The indices `[5, 10, 25]` are just identifiers, and the neural network cannot infer that the words behind indices 5 and 10, for example, may have similar meanings.

This limitation establishes the clear need for the next step: **introducing an Embedding layer**. The embedding will translate these indices into meaning-rich vectors, allowing the neural network to explore the semantic relationships between words and, consequently, drastically improve the model's classification power.
