# 🛒 Amazon Review Sentiment Analysis (TF-IDF + Naive Bayes)

## 📌 Overview

This project performs **sentiment analysis** on Amazon product reviews using:

- Text preprocessing  
- TF-IDF vectorization (Unigrams + Bigrams)  
- Multinomial Naive Bayes classifier  

It also visualizes:

- Class distribution  
- Confusion matrix  
- Most influential positive and negative words  

The model predicts whether a review is:

- Positive  
- Negative  

---

## 🚀 Key Features

✔ Text cleaning using regex  
✔ TF-IDF with 5000 features  
✔ Unigram + Bigram support  
✔ Stratified train-test split  
✔ Accuracy, confusion matrix, classification report  
✔ Word importance visualization  
✔ Custom review prediction function  

---

## 🛠 Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Regex  

---

## 📂 Dataset

The script expects a CSV file:

```
amazon.csv
```

Required columns:

- Text → Review text  
- label → Sentiment (0 = Negative, 1 = Positive)  

---

## 🔎 Project Workflow

### 1️⃣ Text Cleaning

- Convert to lowercase  
- Remove numbers and special characters  
- Keep only alphabetic characters  

```python
clean_text()
```

---

### 2️⃣ Data Visualization

Bar plot showing distribution of:

- Negative reviews  
- Positive reviews  

---

### 3️⃣ Train-Test Split

- 80% training  
- 20% testing  
- Stratified split (maintains class balance)  

---

### 4️⃣ TF-IDF Vectorization

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)
```

- Uses unigrams and bigrams  
- Removes English stopwords  
- Limits features to top 5000  

---

### 5️⃣ Model Training

Model used:

```
Multinomial Naive Bayes
```

Trained on TF-IDF features.

---

### 6️⃣ Evaluation Metrics

- Accuracy  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)  

Confusion matrix is visualized using a heatmap.

---

## 📊 Word Importance Analysis

The model identifies:

- Top 10 words indicating Negative sentiment  
- Top 10 words indicating Positive sentiment  

Based on log probability differences between classes.

Two separate bar charts visualize:

- Negative-indicating words  
- Positive-indicating words  

---

## 🔮 Custom Prediction

You can test any review using:

```python
predict_review("Amazing product, works perfectly!")
predict_review("Very poor quality, broke in a week")
```

Returns:

- "Positive"
- "Negative"

---

## 📦 Installation

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

## ▶️ How to Run

```bash
python your_script_name.py
```

Make sure `amazon.csv` is in the same directory.

---

## 🎯 Use Cases

- Product review analysis  
- Customer feedback monitoring  
- E-commerce analytics  
- NLP beginner projects  
- Text classification research  

---

## 📈 What This Project Demonstrates

- NLP preprocessing  
- TF-IDF feature engineering  
- Probabilistic text classification  
- Model interpretability via word importance  
- Visual performance evaluation  

---

## 👨‍💻 Author

Built as part of Natural Language Processing practice and sentiment analysis experimentation.

If you found this useful, consider starring the repository ⭐
