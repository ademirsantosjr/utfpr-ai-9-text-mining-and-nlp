
# Text Classification with BERT and Transfer Learning

While models like GPT-2 excel at text generation, **BERT (Bidirectional Encoder Representations from Transformers)** is a powerhouse for **language understanding** tasks. Its main feature is the ability to analyze the context of a word by observing the entire text sequence, from both left to right and right to left. This makes it extremely effective for tasks like text classification.

In this guide, we will apply BERT to a **sentiment analysis** problem, classifying product reviews from the e-commerce site Buscapé as positive or negative.

## 1. BERT for Understanding Tasks

BERT uses only the **Encoder** part of the Transformer architecture. During its pre-training, it learns to predict masked (hidden) words in a text and to determine if two sentences are consecutive. This training endows it with a deep contextual understanding of language.

For classification, a classification "head" is added on top of the base BERT model. During **fine-tuning**, this head is trained on our specific dataset, while the BERT weights are adjusted for the new task.

-   **Hugging Face Class:** `TFBertForSequenceClassification` (for TensorFlow)

## 2. Fine-Tuning BERT for Sentiment Analysis

We will fine-tune a pre-trained BERT model for Portuguese, **BERTimbau** (`neuralmind/bert-base-portuguese-cased`), to classify the polarity of product reviews.

### Step 1: Load and Prepare the Data

The process begins in a standard way: we load the dataset and split it into training and test sets.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the reviews dataset
df = pd.read_csv('buscape.csv').dropna()
X = df['review_text'].to_numpy()
y = df['polarity'].to_numpy()

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

### Step 2: Tokenization

We use the `AutoTokenizer` corresponding to the BERTimbau model to convert the text into a format the model understands. We define a maximum sequence length (`max_length=50`) and apply `padding` and `truncation` to ensure all sequences have the same length.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

X_train_encoded = tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=50, return_tensors="tf")
X_test_encoded = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=50, return_tensors="tf")
```

### Step 3: Load, Compile, and Train the Model

We load BERTimbau using `TFBertForSequenceClassification`, which already comes with a classification head ready to be trained. We compile the model with an optimizer and the accuracy metric, and then start fine-tuning with the `fit()` method.

```python
from transformers import TFBertForSequenceClassification

# Load the pre-trained model
model = TFBertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Compile the model
model.compile(optimizer="adam", loss=model.hf_compute_loss, metrics=['accuracy'])

# Perform fine-tuning
model.fit(X_train_encoded, y_train, epochs=5, validation_split=0.1)
```

## 3. Evaluation and Results

After training, we evaluate the model's performance on the test set. The results demonstrate the power of Transfer Learning: with only 5 epochs of training, the model achieves remarkable performance.

```python
from sklearn.metrics import classification_report

# Make predictions on the test set
tf_output = model.predict(X_test_encoded)
y_pred = tf_output.logits.argmax(axis=-1)

# Display the classification report
print(classification_report(y_test, y_pred))
```

**Classification Report:**

```
              precision    recall  f1-score   support

         0.0       0.81      0.62      0.70      1705
         1.0       0.96      0.99      0.97     16702

    accuracy                           0.95     18407
   macro avg       0.89      0.80      0.84     18407
weighted avg       0.95      0.95      0.95     18407
```

### Analysis of the Results

The overall accuracy of **95%** and the F1-score of **0.97** for the majority class (positive) are superior to those obtained with more traditional Machine Learning approaches. This shows that, even with relatively short training, the pre-existing knowledge embedded in BERT about the Portuguese language provides an extremely solid foundation for the classification task.

The computational cost is higher, but the performance gain justifies the use of Transformers for scenarios that require high precision.
