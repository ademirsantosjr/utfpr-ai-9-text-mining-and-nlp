# Advanced Implementation: LSTM and CNN for Text Classification

After establishing a base model and seeing the impact of the Embedding layer, it's time to evolve our architecture. We will replace the intermediate `Dense` layer with more powerful and specialized layers for processing sequences: **LSTM** and **CNN** (`Conv1D`).

## Model with LSTM (Long Short-Term Memory)

LSTM is ideal for capturing context and long-term dependencies in a sequence. It processes embeddings sequentially, maintaining a "memory" of what has already been seen.

### Model Architecture

The main change is the replacement of the `Flatten` and `Dense` layers with a single `LSTM` layer.

1.  `TextVectorization`
2.  `Embedding`
3.  `LSTM` (This layer processes the 3D sequence of embeddings directly)
4.  `Dense` (Output layer)

A notable advantage is that the `LSTM` layer can receive the 3D output from the `Embedding` layer without the need for an intermediate `Flatten` layer.

### Code Example

```python
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

# (Hyperparameters and vectorization layer defined previously)

model_lstm = Sequential([
    vectorization_layer, # Reusing the already adapted layer
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    LSTM(30), # LSTM layer with 30 units
    Dense(1, activation='sigmoid')
])

model_lstm.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

model_lstm.summary()

# Training
# model_lstm.fit(X_train, y_train, epochs=5, validation_split=0.1)
```

### Analysis of the Results

-   **Performance:** Using the `LSTM` layer generally results in a **significant increase in accuracy**, approaching the results of more classic and well-optimized models (such as RandomForest with Bag of Words).
-   **Computational Cost:** Training a model with LSTM is **slower and computationally more expensive**. This is due to the recurrent nature of its processing, which analyzes the sequence step by step.

## Model with CNN (Convolutional Neural Network)

An alternative to LSTM is to use a 1D Convolutional Neural Network (`Conv1D`). In NLP, the CNN acts as a local pattern detector, analyzing "n-grams" (groups of neighboring words) in the sequence of embeddings.

### Model Architecture

The architecture with CNN usually involves:

1.  `TextVectorization`
2.  `Embedding`
3.  `Conv1D` (Applies filters to extract local features)
4.  `MaxPooling1D` (Reduces the dimensionality of the convolution output, keeping the most important features)
5.  `Flatten` (Flattens the output so that it can be processed by a `Dense` layer)
6.  `Dense` (Output layer)

### Code Example

```python
from keras.layers import Conv1D, MaxPooling1D, Flatten

# (Hyperparameters and vectorization layer defined previously)

model_cnn = Sequential([
    vectorization_layer,
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=4),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

model_cnn.summary()

# Training
# model_cnn.fit(X_train, y_train, epochs=5, validation_split=0.1)
```

### Analysis of the Results

The model with `Conv1D` tends to be faster than the `LSTM` and also shows an improvement over the base model with `Dense`. However, for many NLP tasks, the `LSTM` can capture more complex context relationships, resulting in superior accuracy.

## Final Considerations and Next Steps

We have evolved from a simple model to robust Deep Learning architectures. The results can be even better by exploring:

-   **Deeper Architectures:** Stacking multiple `LSTM` or `Conv1D` layers.
-   **Regularization with Dropout:** Adding `Dropout` layers to combat *overfitting*, a particularly important technique in LSTMs.
-   **Hyperparameter Optimization:** Increasing the number of epochs, the dimensionality of the embeddings, the number of neurons/filters, and even using `GridSearch` (with the caveat that the computational cost will be high).

These experiments form the basis for building cutting-edge solutions in Natural Language Processing.
