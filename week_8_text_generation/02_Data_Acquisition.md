
# Corpus Acquisition and Preparation

The first step in building a language model is to define and acquire a **corpus**, which is the set of texts that will serve as the training base. The quality and theme of the corpus are fundamental, as the model will learn and generate texts based on the patterns, vocabulary, and style present in this data.

For this project, our corpus will be composed of a set of **UTFPR undergraduate course regulations**. The goal is for our model to be able to generate texts that follow the formal pattern and terminology found in these documents.

## Data Source

The regulations were extracted from the UTFPR website and pre-processed as HTML files. These files are available in a public repository and can be downloaded to the work environment.

- **Data repository:** [https://github.com/watinha/nlp-text-mining-datasets](https://github.com/watinha/nlp-text-mining-datasets)

## 1. File Download

The first step is to download the HTML files to our local environment. The following script iterates over a list of URLs and downloads each of the regulations.

```python
import io, tarfile, requests, os, pandas as pd

# download the dataset
def download (url, filename=''):
  if (os.path.isfile(filename)):
    print('File already exists in Runtime... All OK')
    return
  response = requests.get(url)
  with open(f'./{filename}', 'wb') as f:
      f.write(response.content)
      print('Download performed and file extracted in Runtime... All OK')

urls = [
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/aa-ab-cf-dispensa-2021.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/ac-2022.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/diretrizes-grad-2022.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/ead-2022.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/estagio-2020.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/extensao-2022.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/rodp-2019.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/tcc-2022.html"
]

filenames = []

for url in urls:
  filename = url.split('/')[-1]
  filenames.append(filename)
  download(url, filename)
```

## 2. Content Extraction with BeautifulSoup

The downloaded files are in HTML format. For our model, we only need the textual content. The **BeautifulSoup** library is an excellent tool for parsing HTML files and extracting the data of interest.

In the script below, we open each file, parse its content, and select only the text contained within `<p>` (paragraph) elements, which is where the main content of the regulations is located. It is important to note the specification of `encoding='cp1252'`, which is necessary to correctly read the special characters present in the original files.

```python
import codecs
from bs4 import BeautifulSoup

corpus = []

for filename in filenames:
  with codecs.open(filename, encoding='cp1252') as f:
    html = f.read()
    soup = BeautifulSoup(html)
    ps = soup.select('div[unselectable=on] ~ p')
    doc = []

    for p in ps:
      doc.append(p.get_text())

    corpus.append('\n'.join(doc))
```

## 3. Text Cleaning and Normalization

The extracted text still contains noise, such as line breaks (`\n`), formatting characters (`\xa0`), and punctuation that can hinder more than help the model learn. The cleaning step aims to normalize the text, making it more consistent and easier to process.

The following function removes punctuation, special characters, and excessive white space, in addition to converting all text to lowercase.

```python
import re

def clean(doc):
  words = doc.split()
  chars_to_replace = '!"#$%&\\'()*+,-:;<=>?@[\]^_`{|}~'
  table = doc.maketrans(chars_to_replace, ' ' * len(chars_to_replace))
  cleaned_words = [w.translate(table) for w in words]
  cleaned_doc = ' '.join(cleaned_words)
  cleaned_doc = cleaned_doc.replace(u'\xa0', u' ')
  cleaned_doc = cleaned_doc.replace(u'\u200b', u' ')
  cleaned_doc = cleaned_doc.replace(u'\n', u' ')
  cleaned_doc = re.sub(r'\s+', ' ', cleaned_doc)
  cleaned_doc = cleaned_doc.lower().lstrip()

  return cleaned_doc


corpus = [ clean(doc) for doc in corpus ]
```

After executing these steps, the `corpus` variable will contain a list of strings, where each string is a clean regulation document ready for the next step: building the training dataset.

```
