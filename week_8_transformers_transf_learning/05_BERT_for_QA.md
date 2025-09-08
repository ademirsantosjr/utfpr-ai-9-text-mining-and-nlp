
# Question Answering (Q&A) with BERT

One of the most impactful applications of Transformer models is **Question Answering (Q&A)**. Specifically, we will explore **Extractive Q&A**, a task where, given a context text and a question, the model locates and extracts the exact snippet of text that contains the answer.

Models like BERT, with their deep contextual understanding capabilities, are ideal for this task, as they are trained to predict the start and end of the answer *span* within the provided context.

## 1. The Simple Approach: Q&A `pipeline`

The most direct way to implement a Q&A system is by using the `pipeline` from the `transformers` library. It abstracts away all the complexity of the process (tokenization, inference, and post-processing), allowing us to focus on the application.

### Workflow

1.  **Instantiate the `pipeline`:** We create a pipeline for the `"question-answering"` task, specifying a pre-trained model for Q&A in Portuguese. Models for this task are often trained on the **SQuAD** (Stanford Question Answering Dataset) benchmark.
2.  **Define Context and Question:** We provide the base text and the question we want to answer.
3.  **Get the Answer:** The pipeline returns a dictionary containing the answer, a confidence score, and the start and end positions of the answer in the text.

```python
from transformers import pipeline

# Load a BERT model trained for Q&A in Portuguese
model_name = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
nlp_qa = pipeline("question-answering", model=model_name, framework='tf')

# Define the context and the question
context = r"""
... (text about the COVID-19 pandemic) ...
"""
question = "When did the Covid-19 pandemic start in the world?"

# Get the answer
result = nlp_qa(question=question, context=context)

print(result)
# Expected output:
# {'score': 0.713..., 'start': 325, 'end': 346, 'answer': 'December 1, 2019'}
```

## 2. Advanced Approach: Retriever-Reader System

The `pipeline` approach works well when we already know which text contains the answer. But what if we have a vast collection of documents (like the UTFPR regulations) and we don't know where the answer is?

The solution is a two-stage system known as **Retriever-Reader**.

### Stage 1: The Retriever

The goal of the Retriever is to filter and find the most relevant documents for the question. Instead of passing all documents to the costly BERT model, we use a faster **Information Retrieval (IR)** technique to create a list of candidates.

A classic and efficient approach for this is **TF-IDF (Term Frequency-Inverse Document Frequency)**, which measures the importance of a word to a document in a collection.

1.  **Vectorize:** We transform all our documents (the regulation articles) into TF-IDF vectors.
2.  **Search:** We transform the question into the same vector space and calculate the similarity (e.g., Euclidean distance) between the question and all documents.
3.  **Rank:** We sort the documents by similarity, obtaining the `top-k` most promising candidates.

### Stage 2: The Reader

The Reader is our BERT Q&A model. It receives the question and only the most relevant documents selected by the Retriever. For each of these documents, it tries to extract an answer and assigns a confidence score.

Finally, the found answers are sorted by their confidence score, and the best answer is presented to the user.

### Limitations

Even with this powerful approach, there are limitations. In one of the tests with the regulations, for the question `"What is the deadline for me to request accompanied activities?"`, the model extracted the answer `"5 days"` with a confidence of 0.49. The answer, although partially correct, is incomplete, as the regulation specifies a deadline of 5 to 45 days. This illustrates that extractive models are literal and may not capture the entirety of complex information.

## Conclusion

Q&A systems based on BERT are powerful tools for extracting precise information from texts. The Hugging Face `pipeline` offers an accessible entry point, while the Retriever-Reader architecture allows us to scale the application to large document bases, creating intelligent and effective search systems.
