# Tokenization and POS Tagger with spaCy

## Accessing Text Components

After processing a text with spaCy, we get a `Document` object. This object contains the text components already processed by the NLP tasks. At this moment, we will focus on the Part-of-Speech (POS) Tagger task and tokenization.

```python
doc = nlp("Now, we are at the beginning of the NLP and Text Mining course.")
```

## Part-of-Speech (POS) Tagger

The POS Tagger, or part-of-speech tagger, is an NLP task that assigns grammatical categories such as nouns, verbs, adjectives, adverbs, among others, to each word in a text.

### Navigating Token Properties

The `Document` object is iterable, which means we can go through its tokens. Each token has a set of properties that can be accessed. To illustrate, let's print some of these properties:

```python
doc = nlp("Now, we are at the beginning of the NLP and Text Mining course.")

for token in doc:
    print(f'{token.text:20}\t {token.tag_:4}\t {token.lemma_:20}\t {token.is_stop}')
```

*   **token.text:** The text of the token.
*   **token.tag_:** The POS Tagger tag.
*   **token.lemma_:** The lemma of the token.
*   **token.is_stop:** Whether the token is a stop word.

When executing this code, we get a table with the following information:

*   **First column (token.text):** The text of each token, including punctuation.
*   **Second column (token.tag_):** The representation of the token considering the POS Tagger. For Portuguese, this is similar to a syntactic analyzer.
*   **Third column (token.lemma_):** The lemma of the token, which is a simplified form of the word, usually its root.
*   **Fourth column (token.is_stop):** A boolean value that indicates whether the word is a stop word.

## Visualization of the Syntactic Structure

spaCy also offers a way to visualize the syntactic structure of the sentence. The `displacy` library can be used to render this visualization. To do this, simply call the `displacy.render` method and pass the `Document` object as a parameter.

**Note:** If you are using a Jupyter Notebook, you need to pass the `jupyter=True` parameter for the visualization to work correctly.
