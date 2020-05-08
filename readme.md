# Exam project in 02462 - Signals and data

## Directory structure (with .gitignore'd files)
```
.
├── 02462_project2020.pdf
├── baseline
│   ├── baseline.ipynb
│   └── glove.6B.50d.txt
├── fasttext
│   ├── fasttext.ipynb
│   └── text_classifier.py
├── __init__.py
├── news_data.npz
├── news_data.zip
├── readme.md
├── spam_data.npz
├── spam_data.zip
├── text_classifier.py
└── utils.py

```

## Articles used
'World' = 'blue' 0, 'sports' = 'red' 1, 'Business'= 'green' 2, 'Sci/Tec'='cyan' 3

formatted as real, could be, text(first paragraph). gathered from frontpage of https://www.bbc.com/news on 08-05-2020

- 0,3/2 https://www.bbc.com/news/business-52392366
- 0 https://www.bbc.com/news/world-us-canada-52584774 
- 0 https://www.bbc.com/news/world-europe-52585162
- 2 https://www.bbc.com/news/business-52570600
- 2 https://www.bbc.com/news/business-52580950 
- 2, 3 https://www.bbc.com/news/business-52570714 
- 3 https://www.bbc.com/news/science-environment-52550973 
- 3 https://www.bbc.com/news/technology-52572381
- 1, 3 https://www.bbc.com/sport/av/athletics/51332721 
- 1 https://www.bbc.com/sport/formula1/52568642
- 1 https://www.bbc.com/sport/boxing/52573766
- 0, 3 https://www.bbc.com/news/uk-england-suffolk-52566082
- 3 https://www.bbc.com/news/science-environment-52560812 

