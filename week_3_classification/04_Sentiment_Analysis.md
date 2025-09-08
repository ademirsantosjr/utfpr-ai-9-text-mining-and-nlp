
# 4. Sentiment Analysis: Classifying Texts with Machine Learning

After reviewing the fundamentals of supervised learning, we can apply them to one of the most common and impactful tasks in Natural Language Processing (NLP): **Sentiment Analysis**.

## What is Sentiment Analysis?

Sentiment Analysis is the process of determining the emotional tone or opinion expressed in a text. The goal is to classify a text as **positive**, **negative**, or, in some cases, **neutral**. This technique is widely used to:

-   Monitor brand reputation on social media.
-   Analyze customer feedback on e-commerce platforms.
-   Evaluate the reception of products, movies, and services.
-   Detect spam in emails.

Given the massive volume of text generated online, manual analysis is impractical. Therefore, we resort to Machine Learning models to automate this task.

## Architecture of a Sentiment Classifier

To build a sentiment analysis system, we follow a well-defined architecture that combines NLP and Machine Learning techniques:

1.  **Input:** The system receives a text as input (e.g., a customer comment, a tweet, a product review).

2.  **Feature Extraction:** Natural language text cannot be directly interpreted by Machine Learning algorithms. We need to convert it into a numerical format. For this, we use the **Bag of Words (BoW)** model:
    -   The BoW represents the text as a set of words, disregarding grammar and order, but keeping the frequency (or count) of each word.
    -   Each unique word in the vocabulary of our dataset becomes a "feature". The value of this feature for a given text is the count of that word in the text.

3.  **Model Training:** The extracted features (the BoW vector) are used to feed a supervised learning algorithm (such as Naive Bayes, SVM, or Logistic Regression). The model is trained with a dataset of texts previously labeled as positive or negative.

4.  **Classification:** Once trained, the model can receive new (unlabeled) texts, extract their features (BoW), and predict the sentiment polarity (positive or negative).

![Sentiment Analysis Architecture](https://i.imgur.com/9e9Z5gG.png) *Simplified diagram of the flow of a sentiment analysis application.*

## Practical Example: Analysis of E-commerce Reviews

To put the theory into practice, we will use a dataset of product reviews from the Buscapé platform, which is publicly available.

### 1. Data Collection and Loading

First, we download and load the dataset, which is in CSV format, using the `pandas` library.

```python
import pandas as pd

# URL of the dataset (example)
url = 'https://raw.githubusercontent.com/some-repo/some-file.csv'

# Loading the data
df = pd.read_csv(url)

print(df.head())
```

### 2. Data Exploration and Cleaning

When analyzing the dataset, we focus on two main columns:

-   `review_text`: Contains the text of the user's review.
-   `polarity`: The sentiment label, where `1` represents a **positive** sentiment and `0` represents a **negative** sentiment.

It is common for real-world datasets to contain inconsistencies. In our case, we may find missing values (NaN - Not a Number) in the polarity column. A simple and effective approach to deal with this is to remove the rows that contain this inconsistent data.

```python
# Remove rows with missing values in the 'polarity' column
df.dropna(subset=['polarity'], inplace=True)

# Check the class distribution
print(df['polarity'].value_counts())
```

An analysis of the distribution may reveal a **class imbalance**, where one class (e.g., positive) has many more examples than the other (e.g., negative). This is something to keep in mind, as it can influence the model's performance.

### Next Steps

With the data cleaned and prepared, the next step is to apply feature extraction with the Bag of Words and, in seguida, treinar e avaliar nosso modelo de classificação de sentimento. This implementation will be detailed in the next document.
