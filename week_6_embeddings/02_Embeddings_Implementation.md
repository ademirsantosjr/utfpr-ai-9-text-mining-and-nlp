
# Embeddings: Implementation and Practical Use with Gensim

Once we understand what Word Embeddings are, the next step is to use them in practical applications. The `gensim` library is a powerful and popular tool in Python for working with topic modeling and word embeddings.

This guide demonstrates how to load and explore pre-trained embedding models using `gensim`.

## 1. Loading a Pre-trained Model

Instead of training our own embeddings from scratch (which requires a massive corpus and a lot of computational power), we can use models that have already been trained by research institutions. In this example, we use a model trained with the **Skip-gram** algorithm on a corpus of Portuguese texts, made available by [NILC](http://www.nilc.icmc.usp.br/nilc/index.php).

To load the model, we use the `KeyedVectors` class from `gensim`.

```python
from gensim.models import KeyedVectors

# Path to the pre-trained embedding file
# (Assuming that the download and extraction have already been done)
embedding_file = 'path/to/your/file/skip_s50.txt'

# Loads the model in word2vec format
model = KeyedVectors.load_word2vec_format(embedding_file)

# Now, the 'model' object contains the vectors of all the words in the vocabulary.
print("Model loaded successfully!")
```

## 2. Exploring the Embeddings Model

With the model in memory, we can begin to explore the semantic relationships it has captured.

### Accessing Vectors

It is possible to access the embedding vector (the numerical representation) of any word in the vocabulary.

```python
# Accesses the vector of the word "king"
vector_king = model['king']

print(f"Vector dimensions: {len(vector_king)}")
# print(f"Vector (first 10 positions): {vector_king[:10]}")
# Expected output: Vector dimensions: 50
```

### Similarity between Words

We can calculate the cosine similarity between the vectors of two words. The result is a value between -1 and 1, where 1 means total identity.

```python
# Calculates the similarity between "elegant" and "beautiful"
similarity = model.similarity('elegant', 'beautiful')
print(f"Similarity between 'elegant' and 'beautiful': {similarity:.2f}")
# Expected output: Similarity between 'elegant' and 'beautiful': 0.80+ (value may vary)

# Compares with an unrelated word
similarity_off = model.similarity('cat', 'fin')
print(f"Similarity between 'cat' and 'fin': {similarity_off:.2f}")
# Expected output: Similarity between 'cat' and 'fin': < 0.30
```

### Finding Nearby Words

One of the most interesting features is asking the model for the most similar words to a specific term.

```python
# Finds the 5 most similar words to "cat"
most_similar_cat = model.most_similar('cat', topn=5)

print("Most similar words to 'cat':")
for word, score in most_similar_cat:
    print(f"- {word} (score: {score:.2f})")

# Expected output may include: dog, puppy, feline, etc.
```

This works because, in the training texts, words like "cat" and "dog" appear in very similar contexts, which brings their vectors closer in the semantic space.

## 3. Vector Arithmetic: The Power of Analogies

As we saw in the introduction, embeddings allow for arithmetic operations. The `most_similar` function can be used to solve analogies, such as the classic "king is to man as queen is to woman".

To do this, we add the "positive" vectors (`king`, `woman`) and subtract the "negative" vector (`man`). The model then looks for the word whose vector is closest to the result.

```python
# Solve the analogy: king - man + woman ≈ queen
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)

print(f"'king' - 'man' + 'woman' ≈ {result[0][0]}")
# Expected output: 'king' - 'man' + 'woman' ≈ queen
```

> **Note**: Although powerful, the result of arithmetic operations can sometimes be inaccurate or unexpected, depending on the training corpus and the complexity of the analogy.

## 4. Comparing Sentences with Word Mover's Distance (WMD)

To measure the semantic "distance" between two entire sentences, a simple average of the word vectors may not be sufficient. `gensim` implements the **Word Mover's Distance (WMD)**, a more robust method.

WMD measures the minimum cost to "transport" the words of one sentence so that its distribution coincides with that of the other sentence. The smaller the distance, the more similar the sentences are.

```python
sentence_1 = "the swimmer quickly left the pool"
sentence_2 = "the swimmer left the pool quickly"
sentence_3 = "john threw the ball up"

# It is necessary to preprocess and tokenize the sentences
sentence_1_tokens = sentence_1.lower().split()
sentence_2_tokens = sentence_2.lower().split()
sentence_3_tokens = sentence_3.lower().split()

# Calculates the WMD distance
distance_1_2 = model.wmdistance(sentence_1_tokens, sentence_2_tokens)
distance_1_3 = model.wmdistance(sentence_1_tokens, sentence_3_tokens)

print(f"Distance between sentences 1 and 2: {distance_1_2:.2f}")
print(f"Distance between sentences 1 and 3: {distance_1_3:.2f}")

# Expected output: the distance between 1 and 2 will be significantly smaller than between 1 and 3.
```

## Conclusion and Next Steps

The `gensim` library offers a set of practical tools for exploring and utilizing the power of word embeddings. The features of similarity, analogy, and sentence comparison open doors to various applications in NLP.

However, for more complex tasks of understanding sentences and documents, more advanced approaches based on Deep Learning, such as Recurrent Neural Networks (RNNs) and Transformers, are often necessary. These will be the next steps in our journey through NLP.
