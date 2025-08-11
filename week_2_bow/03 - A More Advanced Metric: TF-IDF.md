## 3. A More Advanced Metric: TF-IDF

While counting words is useful, some words are more significant than others. Common words like "the" or "a" appear frequently but often carry less meaning than rarer, more specific terms. **TF-IDF (Term Frequency-Inverse Document Frequency)** is a statistical measure that evaluates how important a word is to a document in a collection or corpus.

It is composed of two parts:

### Term Frequency (TF)

**TF measures how frequently a term appears in a document.** It's the ratio of the number of times a term `t` appears in a document `d` to the total number of terms in that document.

**Formula:** `TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)`

The intuition is that the more a word appears in a document, the more important it is *to that specific document*.

### Inverse Document Frequency (IDF)

**IDF measures how important a term is across the entire corpus.** It penalizes common words and gives more weight to words that are rare.

**Formula:** `IDF(t, D) = log( (Total number of documents in corpus D) / (Number of documents containing term t) )`

-   If a term appears in many documents, the ratio inside the `log` approaches 1, and the IDF will be close to 0.
-   If a term is rare, the ratio will be large, resulting in a higher IDF score.

**Implementation Note:** In practice (e.g., in Scikit-Learn), the formula is often smoothed to prevent division by zero and to moderate the weights of very rare terms. A common variant is:
`IDF(t, D) = log( (1 + N) / (1 + df(t)) ) + 1`
where `N` is the total number of documents and `df(t)` is the number of documents containing term `t`.

### The TF-IDF Score

The final TF-IDF score for a word is simply the product of its TF and IDF scores.

**Formula:** `TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)`

This gives a weight that is high when a term appears often in a specific document but rarely in the overall corpus, indicating high relevance for that document.