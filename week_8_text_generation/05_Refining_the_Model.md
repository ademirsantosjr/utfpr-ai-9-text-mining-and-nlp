
# Refining the Model with Transfer Learning and Additional Training

The results of our initial model, trained for a few epochs and with randomly initialized embeddings, were unsatisfactory. The generated text was repetitive and incoherent. In this section, we will explore two main strategies to refine the model and obtain significantly better results: the use of **pre-trained embeddings (Transfer Learning)** and the **increase in the number of training epochs**.

## 1. Transfer Learning with Pre-trained Embeddings

The idea of Transfer Learning is to take advantage of the knowledge captured by models trained on large volumes of data. Instead of training our `Embedding` layer from scratch, we will initialize it with weights from a pre-trained **Word2Vec** model. These embeddings already contain rich semantic representations of words, which accelerates convergence and improves the quality of our model.

### Loading the Embeddings

First, we download the pre-trained vectors (in this case, the 300-dimensional Skip-gram model trained on a large corpus of Brazilian Portuguese) and load them using the `Gensim` library.

### Building the Weight Matrix

Next, we create a **weight matrix**. This matrix will have one row for each word in our model's vocabulary. Each row will contain the 300-dimensional vector corresponding to that word, extracted from the loaded Word2Vec model. If a word from our vocabulary does not exist in the pre-trained model, we initialize its vector with random values.

```python
from gensim.models import KeyedVectors

# Loads the pre-trained vectors
vectors = KeyedVectors.load_word2vec_format(emb_filename)

def get_weight_matrix (vocab, vectors):
  weights_matrix = []
  _, embedding_dim = vectors.vectors.shape

  for word in vocab:
    if word in vectors:
      weights_matrix.append(vectors[word])
    else:
      # If the word does not exist, initialize with a random vector
      weights_matrix.append(np.random.rand(embedding_dim))

  return np.array(weights_matrix, dtype='float32')

# Generates the weight matrix for our vocabulary
weights_matrix = get_weight_matrix(vocab, vectors)
```

### Updating the Model

Now, we define the same model architecture as before, but with a crucial modification in the `Embedding` layer: we initialize it with the weight matrix we just created.

```python
model = Sequential()
# Initializes the Embedding layer with the pre-trained weights
model.add(Embedding(len(vocab), EMBEDDING_DIM, weights=[weights_matrix]))
model.add(LSTM(NEURONS, return_sequences=True))
model.add(LSTM(NEURONS))
model.add(Dense(NEURONS, activation='relu'))
model.add(Dense(len(word_index), activation='softmax'))
```

### Análise dos Resultados com Transfer Learning

After training the model with the pre-initialized embedding layer, the results show a notable improvement. The generated text is less repetitive and the words are more coherent with the context of the regulations. Although it still has errors, the textual structure is visibly superior to that of the initial model.

**Example Output (with pre-trained embeddings):**
> **As complementary activities can be carried out**
> - by the mandatory students of academic relations 01 of extension exchange that technological and extension of the course and the law nº

## 2. The Importance of More Training

Even with Transfer Learning, training for a few epochs is still a limiting factor. Deep language models require extensive training to converge properly. By increasing the number of training epochs (a process that requires more time and computational resources), the model's accuracy increases and the quality of the generated text improves dramatically.

With longer training, the model learns to generate more cohesive sentences, with fewer repetitions and a more correct grammatical structure, replicating the style of the corpus documents more faithfully.

**Example Output (after more training epochs):**
> **The TCC must be supervised by a professor**
> - from utfpr as the support coordinator for one or more professional contr

## 3. Saving and Loading the Final Model

Given the high computational cost of training, it is impractical to train the model every time we want to use it. The solution is to save the trained model and load it when necessary.

-   **Save the Keras Model**: The `model.save()` method stores the architecture, weights, and optimizer configuration in a single file.
-   **Save the `word_index`**: The mapping of words to indices is crucial for decoding the model's output. As it is not saved with the Keras model, we store it separately using the `pickle` library.

```python
import pickle

# Saves the trained model
model.save('09_text_generation.keras')

# Saves the word_index
pickle.dump(word_index, open('09_text_generation_word_index.pkl', 'wb'))
```

To use the model later, simply load it back into memory.

```python
from keras.models import load_model

# Loads the model
model = load_model('./09_text_generation.keras')

# Loads the word_index
word_index = pickle.load(open('./09_text_generation_word_index.pkl', 'rb'))
```

With these refinement techniques, we were able to evolve from a model that generated meaningless texts to a system capable of producing coherent and thematically aligned word sequences to the training corpus.
