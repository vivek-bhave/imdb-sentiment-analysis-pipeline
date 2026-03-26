
# IMDB Sentiment Analysis using Classical Machine Learning

## Objective

The goal of this project is to build a sentiment analysis system using classical machine learning models and understand how feature engineering, model assumptions, and evaluation metrics affect performance on real text data.

This project also focuses on comparing different models and understanding why some models work better than others for TF-IDF based text representation.

The idea is similar to a real scenario where movie producers want to analyze audience reviews to understand satisfaction, complaints, and overall response.

---

## Dataset

IMDB Movie Review Dataset (~40k reviews)

Labels:
- positive
- negative

The dataset is balanced, which helps avoid bias in the model.

```python
df['label'].value_counts()
````

Balanced distribution ensures fair evaluation of precision, recall, and F1-score.

---

## Exploratory Data Analysis

Review length analysis:

```python
df['review_length'] = df['text'].apply(len)
df['review_length'].describe()
```

Observations:

* No missing values
* Right-skewed length distribution
* Very short reviews may not contain enough sentiment information
* Very long reviews may contain mixed opinions or noise

This helps understand why some predictions may fail.

---

## Train-Test Split

```python
X = df['clean_review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

80/20 split used for reproducible evaluation.

---

## Feature Engineering (TF-IDF)

```python
vectorizer = TfidfVectorizer(
    max_features=30000,
    min_df=10,
    max_df=0.9,
    stop_words='english',
    ngram_range=(1,2)
)
```

Parameter tuning was an important part of the project.

Reasons for parameters:

* max_features → reduce noise and computation
* min_df=10 → remove very rare words
* max_df=0.9 → remove extremely common words
* stop_words → remove non-informative words
* ngram_range=(1,2) → capture phrases like "not good"

Parameters were tuned experimentally to get better accuracy (~0.89).

---

## Models Used

The following models were tested:

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest

Model accuracy:

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 0.893    |
| SVM                 | 0.892    |
| Naive Bayes         | 0.872    |
| Random Forest       | 0.852    |
| Decision Tree       | 0.73     |

---

## Model Comparison

Linear models performed better than tree-based models.

Reason:

TF-IDF produces high-dimensional sparse features, where sentiment depends on the combined effect of many words.

Linear models work well in this situation because they combine all features together.

Tree-based models split on one feature at a time, which makes them less suitable for text data.

Observations:

* Logistic Regression performed best because it learns stable weights for words
* SVM also performed well in high-dimensional space
* Naive Bayes worked well but assumes feature independence
* Decision Tree performed worst due to axis-aligned splits
* Random Forest improved over tree but still weaker than linear models

This shows that linear models are usually better for TF-IDF text classification.

---

## Metric Analysis

Accuracy alone is not enough for real-world use.

Confusion matrix example:

```
[[3495 471]
 [385 3649]]
```

Observations:

* Recall important to detect negative reviews
* Precision important to avoid wrong conclusions
* F1-score used to balance precision and recall

Example:

If a producer analyzes reviews, missing negative feedback may lead to wrong decisions.

So multiple metrics must be checked.

---

## Failure Analysis

Some reviews are difficult to classify because sentiment is not always clear.

Examples of difficult cases:

* Mixed opinions in same review
* Sarcasm
* Long reviews with both positive and negative parts
* Very short reviews with little information

Classical models using TF-IDF only see words, not full context,
so mistakes happen in such cases.

---

## Limitation Due to Human Emotion

Binary sentiment (positive / negative) is not always enough.

A review may contain both good and bad opinions.

Example:

* good acting but bad story
* nice music but boring movie

The dataset forces a single label, which makes classification harder.

Sarcasm also causes errors because models rely on words,
not deeper meaning.

A better approach could be:

* multi-class sentiment
* multi-label emotion detection

This could represent human emotions more accurately.

---

## Use Case

Possible applications:

* Movie review analysis
* Product review monitoring
* Customer feedback analysis
* Social media sentiment tracking

The system can automatically classify large numbers of reviews.

---

## Project Goal

The main goal of this project was not only to get accuracy,
but to understand:

* how TF-IDF affects models
* how model assumptions affect results
* why linear models work better for text
* limitations of classical machine learning for sentiment analysis

```

---

```
