
# 5. Implementation of a Sentiment Classifier

With our sentiment analysis dataset prepared, we can now focus on the implementation, training, and evaluation of our classification model. This document details the step-by-step process, using the tools we have already explored.

## Machine Learning Pipeline for Sentiment Analysis

The complete process can be summarized in the following steps:

1.  **Data Preparation:** Separate the text (corpus) and the labels (polarity).
2.  **Train-Test Split:** Use `train_test_split` to create datasets for training and evaluation.
3.  **Feature Extraction (Bag of Words):** Convert the training texts into a numerical matrix using `CountVectorizer`.
4.  **Model Training:** Train a classifier (e.g., Decision Tree) with the training data.
5.  **Model Evaluation:** Use the trained model to predict the sentiments of the test texts and compare them with the real labels to measure accuracy.

### Step 1 and 2: Data Preparation and Splitting

First, we separate the columns of interest from our DataFrame and then split the data.

```python
from sklearn.model_selection import train_test_split

# Assuming 'df' is our clean DataFrame
corpus = df['review_text'].tolist()
labels = df['polarity'].tolist()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)
```

### Step 3: Feature Extraction with `CountVectorizer`

Now, we transform the texts from the training set into numerical vectors. We will use `CountVectorizer`, an implementation of the Bag of Words model.

It is crucial to understand the difference between `fit_transform` and `transform`:

-   `fit_transform(X_train)`: Used **only** on the training set. It learns the vocabulary (the "dictionary" of all terms) from the training data and then transforms this data into a term count matrix.
-   `transform(X_test)`: Used on the test set. It uses the vocabulary **already learned** in the training step to transform the test data. You should not use `fit` or `fit_transform` on the test data, as this would cause "data leakage" and create a feature representation inconsistent with that of the training.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate the vectorizer
vectorizer = CountVectorizer(binary=True) # binary=True indicates presence/absence of the word

# Learn the vocabulary and transform the training data
X_train_features = vectorizer.fit_transform(X_train)

# Only transform the test data with the same vocabulary
X_test_features = vectorizer.transform(X_test)
```

### Step 4 and 5: Training and Evaluation

With the features ready, we can train and evaluate the model.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Instantiate and train the model
model = DecisionTreeClassifier()
model.fit(X_train_features, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_features)

# Evaluate the results
print(classification_report(y_test, y_pred))
```

## Analysis of Results and Considerations

By running this pipeline, we obtain a classification report with metrics such as `precision`, `recall`, and `f1-score`.

### Dimensionality and Sparsity

The Bag of Words model creates a high-dimensionality feature space. The number of features is equal to the number of unique words in the vocabulary (in our case, it can reach tens of thousands). This results in **sparse vectors**, where most of the values are zero, as a single comment contains only a small fraction of the entire vocabulary. This is a striking feature of NLP problems.

### Model Performance and Class Imbalance

An overall accuracy of, for example, 91% may seem good at first glance. However, it is crucial to analyze the performance by class. Frequently, the model may perform very well on the majority class (e.g., positive comments) and poorly on the minority class (e.g., negative comments).

In our case, the identification of negative sentiments may have a significantly lower accuracy. From a business point of view, identifying customer dissatisfaction is often more valuable than confirming satisfaction. Therefore, improving performance on the minority class will be an important focus in future iterations.

### Next Steps

-   **Experiment with different feature extractors:** Instead of `CountVectorizer`, we can use `TfidfVectorizer` to weight the importance of words.
-   **Test different classification algorithms:** Models like `LogisticRegression` or `LinearSVC` usually perform very well on textual data.
-   **Techniques to deal with class imbalance:** Explore methods such as oversampling (increasing the minority class) or undersampling (decreasing the majority class).

This implementation serves as a solid foundation. The next steps will focus on refining each component of this pipeline to build an even more accurate and robust sentiment classifier.
