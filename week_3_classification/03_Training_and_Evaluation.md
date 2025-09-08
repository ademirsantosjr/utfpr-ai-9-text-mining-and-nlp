
# 3. Training and Evaluation of Machine Learning Models

The development of a Machine Learning model is an iterative process that involves two crucial phases: **training** and **evaluation**. Understanding how to separate and use data for these two stages is essential for building robust and reliable models.

## The Training Phase

**Training** is the process of "teaching" the Machine Learning algorithm. In this phase:

1.  **We Choose an Algorithm:** We select a classification model (e.g., Decision Tree, Logistic Regression, etc.).
2.  **We Provide Labeled Data:** We use a dataset where the features and classes (labels) are already known.
3.  **We Execute the Training:** We invoke the model's `fit(X, y)` function, passing the features (`X`) and labels (`y`) of our training dataset.

During training, the algorithm analyzes the data and creates an internal representation of knowledge, establishing the rules and patterns that associate the features with the classes. The result is a **trained model**, ready to be used.

## The Evaluation Phase

After training, we need to verify if the model has learned correctly and if it is capable of **generalizing** to new data, that is, data that was not used in training. This is the **evaluation** phase.

The goal is to use the trained model to make predictions on a new dataset (the **test dataset**) and compare the model's predictions with the real (known) labels of that set.

### The Importance of Separating Data: Train and Test

It is fundamental that the data used to evaluate the model is different from the data used to train it. If we evaluate the model with the same data as the training, we will have an optimistic and unrealistic view of its performance, as the model would just be "remembering" the answers it has already seen.

To facilitate this process, the `scikit-learn` library offers the `train_test_split` function.

#### Using `train_test_split`

This function divides a dataset into two subsets: one for training and another for testing. This ensures that the evaluation is done on "unseen" data for the model.

```python
from sklearn.model_selection import train_test_split

# X: Features of the entire dataset
# y: Labels of the entire dataset

# Split the data: 75% for training, 25% for testing (default)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# X_train: Features for training
# X_test: Features for evaluation
# y_train: Labels for training
# y_test: Labels for evaluation (the "answer key" or ground truth)
```

The workflow becomes:

1.  **Split the data** with `train_test_split`.
2.  **Train the model** using `X_train` and `y_train`: `model.fit(X_train, y_train)`.
3.  **Make predictions** on the test set: `y_pred = model.predict(X_test)`.
4.  **Evaluate the performance** by comparing `y_pred` (the model's predictions) with `y_test` (the real labels).

### Evaluation Metrics

How do we measure how good our model is? We use **evaluation metrics**. Although there are several, three of the most common for classification problems are:

-   **Precision:** Of the times the model predicted a class, how many did it get right? It is a measure of the "quality" of the predictions.
-   **Recall (or Sensitivity):** Of all the instances of a class that actually exist in the dataset, how many did the model manage to identify? It is a measure of "completeness" or "coverage".
-   **F1-Score:** The harmonic mean of Precision and Recall. It is a single metric that seeks a balance between the two.

Analyzing these metrics allows us to understand the strengths and weaknesses of our classifier, identify where it might be failing, and, finally, iterate to build better and more accurate models.
