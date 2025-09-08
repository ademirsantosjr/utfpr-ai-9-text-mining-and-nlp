
# Performance Alternative: Stemming

In the quest to optimize our model, we saw that lemmatization offers a semantically rich dimensionality reduction, but at a high computational cost. When preprocessing time is a critical factor, a faster alternative is necessary: **Stemming**.

## What is Stemming?

Stemming is a heuristic, rule-based process for reducing a word to its root (or *stem*). Unlike lemmatization, stemming is not concerned with the meaning or linguistic validity of the generated root; it simply tries to remove common suffixes and prefixes.

It is a more "crude" approach, but significantly faster.

## Stemming vs. Lemmatization: A Practical Comparison

The best way to understand the difference is by seeing the results side by side. While lemmatization seeks the "infinitive verb" or the "singular noun", stemming just "cuts off" the end of the word.

| Original Word | Lemma (spaCy) | Stem (NLTK) | Observation |
| :--- | :---: | :---: | :--- |
| walking | to walk | `walk` | Stemming creates a root that is not a word. |
| passing | to pass | `pass` | Again, a non-existent root. |
| we are | to be | `we` | The root is short and loses the context of the verb. |
| I went | to be | `I went` | **Stemming Failure**: Fails to reduce an irregular verb. |
| you will say | to say | `you will say` | Both processes can have their limitations. |
| adoption | adoption | `adopt` | **Serious Failure**: Loss of meaning. |
| sweetener | sweetener | `sweeten` | **Over-stemming**: Two different words are reduced to the same root. |

As we can see, stemming is more prone to errors, either by not reducing enough (*under-stemming*) or by being too aggressive and grouping words with different meanings (*over-stemming*).

## Implementation with NLTK

The `spaCy` library focuses on linguistic accuracy and does not offer stemmers. For this task, the `NLTK` (Natural Language Toolkit) library is the most used. For Portuguese, we use the `RSLPStemmer`.

```python
# It is necessary to have NLTK installed
# pip install nltk
import nltk

# Download the necessary resources (only once)
# nltk.download('rslp')

from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()

print(f"'walking' -> {stemmer.stem('walking')}")
print(f"'happiness' -> {stemmer.stem('happiness')}")

# The integration into the pipeline would be similar to that of lemmatization,
# applying the .stem() function to each token in the corpus.
```

## The Trade-off: When to Use Stemming?

The choice between Stemming and Lemmatization is a classic trade-off between speed and accuracy.

**Use Stemming when:**
-   **Processing time** is the most important factor.
-   You are working with **extremely large datasets**, where lemmatization would be prohibitively slow.
-   Small inaccuracies in word grouping are acceptable for your application.

**Use Lemmatization when:**
-   **Linguistic accuracy** is essential.
-   The result of the processing needs to be interpretable by humans (lemmas are real words, stems are not).
-   You can afford the computational cost of a slower preprocessing, which is executed only once.

## Conclusion of the Week

Throughout this journey, we explored an iterative workflow to improve a text classification model:

1.  **Baseline**: We started with a simple Bag-of-Words model.
2.  **Stopword Removal**: An easy and low-cost optimization that cleaned up the initial noise.
3.  **Lemmatization/Stemming**: Powerful techniques to reduce dimensionality by grouping words. The choice between them depends on your use case and available resources.
4.  **N-Grams**: The most impactful strategy for performance, which reintroduced the context of word order at the cost of a large increase in the number of features.

Building an effective NLP system is a process of experimentation, where each technique is a tool to be applied and measured. The next step is to combine these tools into a robust `Pipeline` and use `GridSearch` techniques to find the combination of parameters that delivers the best result for your specific problem.
