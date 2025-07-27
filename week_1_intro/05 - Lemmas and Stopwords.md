# Lemmas and Stop Words with spaCy

## Introduction

In this section, we will cover the tasks of identifying lemmas and stop words, which are fundamental in text pre-processing for text mining and machine learning applications.

## Lemmatization

Lemmatization is an NLP task that consists of reducing inflected words to their base or lemma form, that is, the form in which they appear in a dictionary. The main objective of lemmatization is to reduce the dimensionality of problems by grouping different forms of the same word into a single representation.

### Accessing Lemmas

Lemmas can be accessed through the `lemma_` property of each token. For example, a verb like "tem" (has) is reduced to its infinitive form "ter" (to have). Words that are a junction of a preposition with an article, such as "na" (in the), are broken down into their constituent parts, such as "em" (in) and "a" (the).

## Stop Words

Stop words are common words that are often removed from a text in text mining applications because they do not contribute significantly to identifying the meaning of the text. Examples of stop words in Portuguese include prepositions, definite and indefinite articles, and other terms that do not add much semantic value.

### Identifying Stop Words

spaCy has a specific list of stop words for each language. The `is_stop` property of each token is a boolean value that indicates whether the word is a stop word or not. Removing stop words helps to reduce noise and improve the efficiency of machine learning algorithms.

## Applications of NLP Tasks

NLP tasks have several practical applications:

*   **POS-Tagger:** Used in spell checkers and in the construction of text meaning graphs.
*   **Named Entities:** Used on news sites to generate automatic links between news items and in the extraction of information such as prices and quotes.
*   **Lemmatization, Stemming, and Stop Words:** Mainly used for dimensionality reduction in machine learning models.