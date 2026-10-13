# IMDb Sentiment Analysis Using Classical Machine Learning

A machine learning application for analyzing audience sentiment in movie reviews, comparing classical classification algorithms, and extracting interpretable feedback signals to support audience-insight analysis.

### Live Application

**Try the deployed application:** https://movie-sentiment-analyzer-v34h.onrender.com/

You can:
- Explore the included sample movie-review dataset.
- Download the sample CSV.
- Upload your own CSV containing a `content` column.
- Compare positive and negative review rankings.
- Explore influential words through the automatic analysis mode.


## 1. Problem Statement

Movie producers and content stakeholders receive large volumes of audience reviews. Manually examining these reviews to understand audience satisfaction, dissatisfaction, and recurring complaints can be time-consuming and difficult to scale.

A simple positive/negative classification system can help organize feedback, but classification alone does not explain which reviews deserve attention or which words contribute to the predicted sentiment.

This project addresses the problem by developing a classical machine learning pipeline that classifies movie reviews, ranks reviews according to predicted sentiment, and extracts influential words from selected reviews to help identify potential feedback themes.

The project also investigates how text feature engineering, model assumptions, and evaluation metrics influence classification performance.

## 2. User Requirements

The system is designed around the following stakeholder requirements.

### Functional Requirements

- **Sentiment-based review selection:** Identify reviews with the strongest predicted positive or negative sentiment.
- **Automatic feedback exploration:** Extract influential words from selected reviews and map recognized terms to human-readable feedback categories.
- **Flexible data input:** Allow users to analyze an included sample dataset or upload their own CSV file.
- **Interactive interface:** Provide a web interface through which users can select the target sentiment and analysis mode.
- **Accessible deployment:** Make the application available through a public web URL for demonstration and evaluation.

### User Workflows

| Mode | Purpose | Output |
|---|---|---|
| Manual | Explore the strongest reviews for a selected sentiment | Up to five ranked reviews |
| Automatic | Examine influential words from selected reviews | Extracted words and mapped feedback categories |
| Sample dataset | Test the application without preparing a dataset | Analysis of included sample reviews |
| CSV upload | Analyze a user's own movie-review collection | Results based on the uploaded reviews |

The automatic mode is intended to support exploratory feedback analysis. Its mapped categories are based on predefined word mappings, not a separately trained topic-modeling system.

## 3. Project Objectives

The project focuses on four primary objectives:

1. Understand how text preprocessing and TF-IDF feature engineering transform unstructured reviews into numerical representations.
2. Compare classical machine learning algorithms on high-dimensional, sparse text data.
3. Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix analysis.
4. Translate the trained model into an interactive application that supports practical review exploration.

## 4. Dataset

The project uses an IMDb movie-review dataset containing approximately 40,000 reviews with binary sentiment labels.

- **Positive:** Reviews expressing favorable opinions.
- **Negative:** Reviews expressing unfavorable opinions.

The dataset is described as balanced, supporting evaluation of both sentiment classes without a substantial class-frequency imbalance.

### Exploratory Data Analysis

Review length was examined to understand the characteristics of the text data.

```python
df['review_length'] = df['text'].apply(len)
df['review_length'].describe()
```

Key observations:

- No missing values were observed in the analyzed dataset.
- Review lengths followed a right-skewed distribution.
- Very short reviews may contain insufficient sentiment information.
- Very long reviews may contain mixed opinions, irrelevant text, or conflicting sentiment signals.

These characteristics help explain why certain reviews may be more difficult to classify than others.

## 5. Data Preparation

The text data was cleaned before feature extraction, and the dataset was divided into training and testing subsets.

```python
X = df['clean_review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

An 80/20 train-test split was used with a fixed random state to support reproducible evaluation.

## 6. Feature Engineering: TF-IDF

Text cannot be directly processed by conventional scikit-learn classification algorithms. The reviews were therefore transformed into numerical feature vectors using Term Frequency–Inverse Document Frequency (TF-IDF).

The vectorizer configuration was:

```python
TfidfVectorizer(
    max_features=30000,
    min_df=10,
    max_df=0.9,
    stop_words='english',
    ngram_range=(1, 2)
)
```

### Feature Engineering Decisions

| Parameter | Purpose |
|---|---|
| `max_features=30000` | Limit the vocabulary size and control computational cost |
| `min_df=10` | Exclude terms appearing in fewer than ten documents |
| `max_df=0.9` | Exclude terms appearing in more than 90% of documents |
| `stop_words='english'` | Remove common English stopwords |
| `ngram_range=(1, 2)` | Represent individual words and two-word phrases |

Including bigrams allows the model to represent phrases such as *"not good"* rather than treating every word only as an independent token.

The parameters were selected experimentally, with the objective of improving classification performance while controlling vocabulary size and reducing less informative features.

## 7. Classical Machine Learning Models

Five classification algorithms were evaluated using TF-IDF text representations.

| Model | Accuracy |
|---|---:|
| Logistic Regression | 89.3% |
| Support Vector Machine (SVM) | 89.2% |
| Naive Bayes | 87.2% |
| Random Forest | 85.2% |
| Decision Tree | 73.0% |

Logistic Regression achieved the highest observed accuracy of approximately 89.3%, marginally outperforming SVM.

### Model Comparison and Interpretation

**Logistic Regression**

Achieved the highest accuracy in the experiment. It learns feature weights that combine evidence from many words to estimate the probability of each sentiment class. Its learned coefficients also support analysis of influential vocabulary.

**Support Vector Machine**

Performed almost identically to Logistic Regression. SVM is well suited to high-dimensional feature spaces and can establish an effective decision boundary between sentiment classes.

**Naive Bayes**

Provided competitive performance despite its conditional-independence assumption. In natural language, words are not truly independent, and word combinations can carry important sentiment information.

**Random Forest**

Outperformed a single Decision Tree but remained below the linear models. Its ensemble structure can improve stability, although tree-based splitting may be less effective when sentiment depends on many sparse text features.

**Decision Tree**

Achieved the lowest accuracy. Individual feature-based splits can struggle to represent sentiment patterns distributed across a large vocabulary.

### Key Finding

Linear models performed better than the tree-based models in this experiment.

TF-IDF produces high-dimensional, sparse representations in which the combined contribution of many words can determine sentiment. Linear models can effectively combine these signals, while tree-based methods rely on feature-based splits.

The results demonstrate why algorithm selection should consider the representation and structure of the input data rather than relying solely on the popularity of a model.

## 8. Model Evaluation and Metric Analysis

Accuracy alone does not establish whether a sentiment classifier is suitable for a practical feedback-analysis workflow.

The evaluation therefore considered precision, recall, F1-score, and the confusion matrix.

### Confusion Matrix

The reported confusion matrix was:

```text
[[3495,  471],
 [ 385, 3649]]
```

The class ordering used in the project report is negative followed by positive.

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 3,495 | 471 |
| Positive | 385 | 3,649 |

This reveals two important types of error:

- **False negatives:** Actual negative reviews classified as positive.
- **False positives:** Actual positive reviews classified as negative.

### Why Recall Matters

For a producer investigating audience dissatisfaction, failing to identify a negative review can conceal a genuine complaint.

The reported negative-class recall was approximately 0.88, meaning that around 88% of actual negative reviews were correctly identified in the evaluated results.

### Why Precision Matters

The reported negative-class precision was approximately 0.90. This means that approximately 90% of reviews predicted as negative were actually negative under the evaluation labels.

Incorrectly including positive reviews in a negative-feedback analysis can lead stakeholders to draw misleading conclusions about audience dissatisfaction.

### Why F1-Score Matters

F1-score balances precision and recall. It is useful when both missed complaints and incorrectly flagged reviews matter.

For practical feedback analysis, these metrics provide complementary information that cannot be captured by accuracy alone.

## 9. Failure Analysis

The project examines why a model with relatively high accuracy can still misclassify individual reviews.

### Mixed Sentiment

A review might express satisfaction with one aspect of a movie while criticizing another.

Example: "The acting was excellent, but the story was disappointing."

A binary classifier must assign a single sentiment label even when the review contains conflicting opinions.

### Sarcasm and Indirect Expressions

A review can contain positive words while communicating a negative opinion through sarcasm or context.

A TF-IDF-based classifier primarily learns statistical associations between features and labels; it does not inherently understand the intended meaning of an expression.

### Review Length

Very short reviews may lack sufficient information. Longer reviews may contain several opinions, making a single overall sentiment difficult to infer from individual word contributions.

### Context and Word Combinations

The meaning of a word can depend on surrounding words. Although bigrams capture some local word combinations, TF-IDF does not provide a complete representation of linguistic context.

These limitations explain why a model can perform well overall while remaining unreliable for certain types of reviews.

## 10. Limitations of Binary Sentiment Classification

Human opinions are not always strictly positive or negative.

Consider the following examples:

- Good acting but a bad story.
- Nice music but a boring movie.
- Strong performances but disappointing pacing.

A single sentiment label cannot fully represent these different aspects of an opinion.

Sarcasm and indirect expressions introduce further ambiguity because surface-level word patterns may not reflect the intended sentiment.

Potential extensions include:

- **Multi-class sentiment classification:** Represent additional sentiment categories.
- **Aspect-based sentiment analysis:** Analyze sentiment toward specific movie attributes, such as acting, story, pacing, or cinematography.
- **Multi-label emotion detection:** Allow a review to express multiple emotions rather than forcing it into a single category.

These approaches could provide a more detailed representation of audience feedback.

## 11. Application Architecture and Implementation

The trained Logistic Regression model was integrated into a Flask web application.

### Processing Pipeline

```text
IMDb Reviews / Uploaded CSV
            |
            v
      Input Validation
            |
            v
       Text Cleaning
            |
            v
      TF-IDF Vectorizer
            |
            v
   Logistic Regression Model
            |
            v
   Positive / Negative Scores
            |
            v
       User Selection
        /          \
       v            v
 Manual Ranking   Automatic Analysis
       |            |
       v            v
 Top 5 Reviews   Influential Words
                     |
                     v
               Semantic Mapping
                     |
                     v
               Display Results
```

### Implementation Details

**Flask:** Provides the web interface and handles user submissions.

**Pandas:** Reads uploaded CSV files and processes review data.

**Scikit-learn:** Provides the TF-IDF vectorizer, trained classifier, and feature representations.

**NumPy:** Supports feature-contribution calculations and ranking.

**Pickle:** Loads the saved Logistic Regression model and TF-IDF vectorizer.

**HTML and CSS:** Provide the user interface for sample-data exploration, file uploads, sentiment selection, and results display.

### Interpretable Word Contributions

For the Logistic Regression model, the contribution of a feature can be examined using its TF-IDF value and learned model coefficient.

For a feature \(j\), the contribution is:

\[
c_j = x_j w_j
\]

where \(x_j\) is the feature value and \(w_j\) is the learned coefficient.

Positive and negative contributions can be ranked to identify influential vocabulary for a review.

The application uses these contributions to extract words from selected reviews. It then applies predefined mappings to convert recognized terms into human-readable feedback categories.

This is a lightweight, model-based explanation mechanism. It is not a guarantee of causal explanation, and the mapped categories depend on the vocabulary and rules defined in the application.

## 12. Deployment

The application is structured for deployment as a Flask web service on Render.

The production server uses Gunicorn, while the trained model and vectorizer are loaded from saved files rather than retrained for each request.

### Deployment Configuration

| Component | Configuration |
|---|---|
| Web framework | Flask |
| Production server | Gunicorn |
| Model artifact | `model_LR.pkl` |
| Vectorizer artifact | `vectorizer.pkl` |
| Sample dataset | `sample_reviews.csv` |
| Frontend | HTML and CSS |
| Hosting platform | Render |

### Run Locally

From the repository root, create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r imdb_app/requirements.txt
```

Start the Flask application:

```bash
python imdb_app/app.py
```

Open the application at:

```text
http://127.0.0.1:5000
```

### Production Start Command

When the Render service uses `imdb_app` as its root directory, the start command is:

```bash
gunicorn app:app
```

The sample dataset and both serialized model artifacts must be available in the deployed application directory.

## 13. Repository Structure

```text
imdb-sentiment-analysis-pipeline/
|
|-- imdb_app/
|   |-- app.py
|   |-- model_LR.pkl
|   |-- vectorizer.pkl
|   |-- sample_reviews.csv
|   |-- requirements.txt
|   |
|   |-- templates/
|       |-- index.html
|
|-- Notebooks/
|   |-- 01_eda_imdb.ipynb
|   |-- 02_model_training.ipynb
|
|-- Reports/
|   |-- Failure_Analysis_and_sentiment_Limitations.pdf
|   |-- Metric analysis.txt
|
|-- README.md
|-- .gitignore
```

## 14. Practical Applications

The underlying approach can be adapted to several feedback-analysis problems:

- Movie review and audience-response analysis.
- Product review monitoring.
- Customer feedback analysis.
- Social media sentiment tracking.
- Identification of potentially negative customer experiences.

Applying the pipeline to another domain would require evaluating the vocabulary, labels, and model performance on data representative of that domain.

## 15. Key Learnings

This project focuses on understanding the complete classical ML workflow rather than optimizing only for accuracy.

Key learnings include:

- How TF-IDF parameters influence vocabulary selection and text representation.
- Why linear classifiers can outperform tree-based models on high-dimensional sparse text.
- How model assumptions affect classification performance.
- Why precision, recall, F1-score, and confusion matrix analysis matter for real-world interpretation.
- How failure analysis exposes limitations that aggregate metrics can hide.
- How a trained model can be integrated into an interactive application for practical use.
- Why binary sentiment labels and word-level features cannot fully represent the complexity of human opinions.

## Conclusion

The project demonstrates a classical machine learning pipeline that combines text preprocessing, TF-IDF feature engineering, model comparison, metric analysis, and application development.

Logistic Regression achieved 89.3% accuracy in the reported experiment, with SVM close behind at 89.2%. The results highlight the effectiveness of linear models for this TF-IDF representation while exposing the limitations of word-based binary sentiment classification.

The final Flask application makes the trained model accessible through a sample dataset and user-provided CSV files, enabling sentiment-based review ranking and exploratory feedback analysis.

The broader objective is to understand not only how a model predicts sentiment, but also why its representation and assumptions influence performance, where its errors arise, and how its predictions can support a practical user workflow.
```