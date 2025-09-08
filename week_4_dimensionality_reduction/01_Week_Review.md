
# Improving Text Classification with Bag-of-Words

In this series of studies, our goal is to enhance text classification in Portuguese, focusing on **dimensionality reduction** strategies and the use of **N-grams** to enrich the Bag-of-Words (BoW) model.

## Context: Sentiment Analysis of E-commerce Reviews

We will use a sentiment analysis dataset based on user comments from the e-commerce platform Buscapé. The objective is to classify a comment as having **positive (1)** or **negative (0)** polarity.

A central challenge in this dataset is its **imbalance**:
- **Positive Comments:** 66,817 samples
- **Negative Comments:** 6,810 samples

This disparity directly impacts the performance of the classification model, which tends to favor the majority class (positive), resulting in low accuracy for the minority class (negative).

## Review of the Classification Pipeline with Bag-of-Words

The standard document classification process we have followed so far can be summarized in the following steps:

1.  **Data Loading**: Reading the dataset.
2.  **Simple Preprocessing**: Initial cleaning, such as removing missing values (`NaN`).
3.  **Corpus and Label Extraction**: Separating the texts (corpus) and their respective classifications (labels).
4.  **Train-Test Split**: Separating the data to train and validate the model impartially.
5.  **Feature Extraction (Train)**: Using `CountVectorizer` to create the term dictionary and transform the training texts into numerical vectors (BoW).
6.  **Model Training**: Fitting a classification algorithm (e.g., Decision Tree) to the vectorized training data.
7.  **Feature Extraction (Test)**: Applying the same `CountVectorizer` (already trained) to transform the test texts into vectors.
8.  **Prediction and Evaluation**: Using the trained model to predict the labels of the test set and evaluate its performance (e.g., accuracy).

## Optimizing the Experimentation Process

Since our goal is to test various configurations (different preprocessing, feature extractors, and classifiers), repeatedly executing steps 5 to 8 can be inefficient.

To facilitate experimentation, we encapsulate this logic into a reusable function: `train_and_validate`.

### `train_and_validate` Function

This function centralizes the steps of feature extraction, training, prediction, and validation.

**Function Signature:**

```python
def train_and_validate(corpus_train, y_train, corpus_test, y_test, feature_extractor, classifier):
    # 1. Extracts features from the training corpus
    X_train = feature_extractor.fit_transform(corpus_train)

    # 2. Trains the classifier
    classifier.fit(X_train, y_train)

    # 3. Extracts features from the test corpus
    X_test = feature_extractor.transform(corpus_test)

    # 4. Makes a prediction and evaluates the accuracy
    accuracy = classifier.score(X_test, y_test)
    print(f"Model accuracy: {accuracy}")

    # (Optional) Displays a detailed classification report
    # from sklearn.metrics import classification_report
    # y_pred = classifier.predict(X_test)
    # print(classification_report(y_test, y_pred))

    return accuracy
```

With this structure, we can easily invoke the complete pipeline, changing only the components we want to test, such as the `feature_extractor` or the `classifier`.

## Next Steps

In the following sections, we will use this function to evaluate the impact of different preprocessing techniques on the performance of our sentiment classification model:
- **Stopword Removal**
- **Lemmatization**
- **Stemming**
- **Using N-grams**
