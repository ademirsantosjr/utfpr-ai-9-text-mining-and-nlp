
# Model Implementation and Training

With the dataset properly structured, we can proceed to the stage of building and training our deep neural network model using the Keras framework.

## 1. Model Architecture

The chosen architecture is a more complex sequential network than the one used in binary classification problems, aiming to capture the nuances of the language for text generation. The layers are stacked as follows:

1.  **`TextVectorization`**: The model's entry point. Transforms the text sequences from our dataset `X` into integer sequences.

2.  **`Embedding`**: Converts integer sequences into dense vector representations (embeddings). In this first version, the layer will be trained from scratch along with the rest of the model, using a dimensionality of 300 for the vectors.

3.  **`LSTM` (two layers)**: The core of our sequential learning model. We use two stacked `LSTM` layers, each with 300 neurons, to learn complex patterns and long-term dependencies in the text.
    -   `return_sequences=True`: This parameter in the first `LSTM` layer is essential. It ensures that the layer returns the complete sequence of outputs, and not just the output of the last time step, which is a requirement for stacking recurrent layers.

4.  **`Dense`**: A fully connected layer with 300 neurons and `relu` activation, which serves to add more learning capacity to the model.

5.  **`Dense` Output (Softmax)**: The final layer, responsible for generating the prediction. It has a number of neurons equal to the size of our output vocabulary and uses the `softmax` activation function to generate a probability distribution over all possible words.

The code below defines the model architecture:

```python
from keras.layers import Embedding, LSTM, Dense, TextVectorization
from keras.models import Sequential
from keras.optimizers import AdamW

VOCAB_SIZE = 20000
MAX_SEQUENCE_SIZE = window_size
NEURONS = 300
EPOCHS = 5
EMBEDDING_DIM = 300

vectorization_layer = TextVectorization(
    VOCAB_SIZE, output_sequence_length=MAX_SEQUENCE_SIZE)
vectorization_layer.adapt(X)

model = Sequential()
model.add(vectorization_layer)
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM))
model.add(LSTM(NEURONS, return_sequences=True))
model.add(LSTM(NEURONS))
model.add(Dense(NEURONS, activation='relu'))
model.add(Dense(len(word_index), activation='softmax'))
```

## 2. Compilation

Before training, the model needs to be compiled. In this step, we define:

-   **Optimizer**: We use `AdamW`, a robust variation of the Adam optimizer.
-   **Loss Function**: `categorical_crossentropy`. This is the default choice for multi-class classification problems, which is how we are treating the prediction of the next word (each word in the vocabulary is a class).
-   **Metrics**: We monitor `accuracy` to evaluate the model's performance during training.

```python
model.compile(optimizer=AdamW(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
```

## 3. Training

Training is started with the `fit` method, passing the input data `X` and the labels `y` (already in one-hot encoded format).

```python
model.fit(X, y, epochs=EPOCHS, validation_split=0.1)
```

It is expected that the training time per epoch will be longer than in simpler models, due to the depth of the network (two LSTM layers) and the volume of the dataset. After 5 epochs, the training accuracy reaches relatively low values (for example, ~17%). This is common in text generation problems, where the model needs to choose the correct word from thousands of options, a task inherently more complex than binary classification.

## 4. Text Generation Function

To test the model, we create a function that receives an initial text sequence (seed) and generates a continuation. The function is not limited to predicting the most likely word; instead, it identifies the **5 most likely words** and generates 5 distinct text continuations, allowing for a richer analysis of the model's capabilities.

```python
def generate_text (model, input, num_words, max_sequence_size, word_index):
  outcomes = []

  generated_words = []
  context = input.split()

  diff = max_sequence_size - len(context)
  initial_context = ['' for i in range(diff)] + context[-max_sequence_size:]
  x_test = ' '.join(initial_context).lstrip()

  pred = model.predict(np.array([x_test], dtype="object"), verbose=0)
  most_probable = [ word_index[i] for i in np.argsort(pred[0])[-5:] ]

  print(input)

  for next in most_probable:
    generated_words = [next]
    context = initial_context[1:]
    context.append(next)

    for i in range(num_words):
      x_test = ' '.join(context).lstrip()

      pred = model.predict(np.array([x_test], dtype="object"), verbose=0)
      next_word = word_index[np.argmax(pred[0])]
      generated_words.append(next_word)
      context = context[1:]
      context.append(next_word)

    print(' - ' + ' '.join(generated_words))
```

## 5. Analysis of Initial Results

When testing the model trained for only 5 epochs, the results are unsatisfactory. The generated text is largely incoherent and repetitive, often consisting of punctuation marks or disconnected words.

**Example Output:**

> **Validation is a procedure for students to be able to**
>  - aaes of completed set of extension notices of extension of undergraduate courses and from being of being

This poor result is a direct reflection of insufficient training. The model has not yet learned the complex language patterns present in the corpus. This establishes the need to refine our approach, which will be the focus of the next section.
