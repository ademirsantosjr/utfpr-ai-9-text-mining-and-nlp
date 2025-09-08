# From the Bag of Words Approach to Sequences

When studying the *Bag of Words (BoW)* model, we notice some important limitations that impact the model's ability to understand natural language more deeply:

- **Loss of Word Order:** The sentence structure and context are lost, as the model treats the text as a disordered set of words.
- **Sparsity and High Dimensionality:** It generates very large and sparse vectors (full of zeros), which can be computationally inefficient.
- **Lack of Semantic Analysis:** The meaning and nuances of words are not captured.

The use of **sequences** directly addresses the issues of order loss and sparsity, paving the way for more sophisticated models.

## The Sequence Model

Advanced document classification models use sequences to preserve the original order of words. This approach is the basis for the use of embeddings, deep neural networks, and architectures like *Transformers*.

### Construction Process

The process of creating sequences from a text is similar to the beginning of vectorization with BoW, but with a different purpose:

1.  **Vocabulary Creation:** As with BoW, the first step is to build a vocabulary (or dictionary) with all the unique tokens present in the training corpus.

2.  **Index Assignment:** Each token in the vocabulary receives a unique integer index. This index will be its representation.

3.  **Sequence Construction:** Each text document is transformed into a vector (sequence) of integer indices, where each index corresponds to a token, respecting the order in which it appears in the text.

**Practical Example:**

Consider the sentence: `"John won the race."`

- The vocabulary could map: `{"john": 2, "won": 3, "the": 4, "race": 5, ".": 6}`
- The resulting sequence would be: `[2, 3, 4, 5, 6]`

## Essential Treatments in Sequences

For sequences to be used by Deep Learning models, some pre-processing steps are crucial.

### Padding

Deep learning models require that the inputs (our sequences) have a **fixed size**. However, texts naturally have varying lengths. To solve this, we apply **padding**:

- We define a maximum length for the sequences.
- Sequences shorter than the maximum are filled with a specific value (usually 0) until they reach the defined length.
- Longer sequences are truncated.

This ensures that all input vectors have the same dimension.

### Handling Unknown Words (UNK)

During inference (when the already trained model is used on new data), it is possible that words that were not in the original vocabulary may appear. To handle this, we reserve a specific index for a special token, usually called **UNK** (for *Unknown*).

Any new word found is mapped to this index, allowing the model to process the information without breaking, even if it does not know the specific token.

### The Importance of Punctuation

Unlike in the *Bag of Words* model, where punctuation is often discarded, in sequence models it can be kept as a valid token. Punctuation carries structural and semantic information (think of the difference between a question and a statement) and can be useful for the model.
