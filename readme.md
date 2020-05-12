# Exam project in 02462 - Signals and data
This repository contains project work in the course "Signals and data" taught at the Technical University of Denmark. It serves as experiments for a comparison between two text classification methods.

The project is divided into two notebooks. Common functions are found in `common.py`.

The experiments for each methods can be found in in the notebook in their respective folders.

## Directory structure 
```
.
├── baseline
│   ├── baseline.ipynb
│   └── glove.6B.50d.txt
├── common.py
├── emails.csv
├── fasttext
│   ├── CV
│   ├── fasttext.ipynb
│   ├── news_fasttext_classifier.p
│   ├── news_train_emb.p
│   ├── spam_fasttext_classifier.p
│   ├── spam_train_emb.p
│   └── text_classifier.py 
├── __init__.py
├── news_data.npz
├── news_data.zip
├── readme.md
├── similar_news.txt
├── spam_data.npz
└── spam_data.zip
```

## Articles used
'World' = 'blue' 0, 'sports' = 'red' 1, 'Business'= 'green' 2, 'Sci/Tec'='cyan' 3

formatted as real, could be, text(first paragraph). gathered from frontpage of https://www.bbc.com/news on 08-05-2020

- 0,3/2 https://www.bbc.com/news/business-52392366
- 0 https://www.bbc.com/news/world-us-canada-52584774 
- 0 https://www.bbc.com/news/world-europe-52585162
- 0, 3 https://www.bbc.com/news/uk-england-suffolk-52566082
- 1, 3 https://www.bbc.com/sport/av/athletics/51332721 
- 1 https://www.bbc.com/sport/formula1/52568642
- 1 https://www.bbc.com/sport/boxing/52573766
- 2 https://www.bbc.com/news/business-52570600
- 2 https://www.bbc.com/news/business-52580950 
- 2, 3 https://www.bbc.com/news/business-52570714 
- 3 https://www.bbc.com/news/science-environment-52550973 
- 3 https://www.bbc.com/news/technology-52572381
- 3 https://www.bbc.com/news/science-environment-52560812 
