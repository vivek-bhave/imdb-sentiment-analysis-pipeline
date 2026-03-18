# IMDB Sentiment Analysis using Classical ML

## Objective

Build a sentiment analysis system using classical machine learning
and understand how feature engineering, model assumptions, and
evaluation metrics affect real-world performance.

The project simulates a real scenario where movie producers want to
analyze audience reviews to understand satisfaction and complaints.

---

## Dataset

IMDB Movie Review dataset (~40k reviews)

Labels:
- positive
- negative

Dataset is balanced, which helps avoid model bias.

```python
df['label'].value_counts()

Balanced distribution ensures fair evaluation of precision and recall.

Exploratory Data Analysis

Review length analysis:

df['review_length'] = df['text'].apply(len)
df['review_length'].describe()

Observations:

No missing values

Right-skewed length distribution

Very short reviews may lack sentiment information

Very long reviews may contain mixed sentiment or noise

This helps understand possible model errors.

Train-Test Split
X = df['clean_review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

80/20 split used for reproducible evaluation.

Feature Engineering (TF-IDF)
vectorizer = TfidfVectorizer(
    max_features=30000,
    min_df=10,
    max_df=0.9,
    stop_words='english',
    ngram_range=(1,2)
)
Parameter tuning (Important part of project)

max_features → limit noise & computation

min_df=10 → remove rare noisy words

max_df=0.9 → remove very common words

stop_words → remove non-informative words

bigrams → capture phrases like "not good"

Parameters were tuned experimentally to improve accuracy (~0.89).

Model

Logistic Regression used as baseline.

Reason:

Works well with high-dimensional sparse TF-IDF data

Linear models perform well for text classification

Metric Analysis (Not only accuracy)

Accuracy alone is not enough for real-world use.

Confusion matrix:

[[3495 471]
 [385 3649]]

Observations:

Recall important to detect negative reviews

Precision important to avoid wrong conclusions

F1 used to balance precision and recall

Example use case:
Producer analyzing reviews must not miss negative feedback.

Detailed metric analysis included in project notes.

Use Case

Possible applications:

Movie review analysis

Product review monitoring

Customer feedback analysis

Social media sentiment tracking

System can help filter positive/negative reviews automatically.

Pending Work

Failure analysis of misclassified reviews

Compare more models (SVM, Naive Bayes, RF)

Deployment (API)

Database logging

Goal: understand limitations of classical sentiment models.