# SMS-Spam-Prediction

📩 SMS Spam Prediction

A machine learning project to classify SMS messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) techniques.

🧠 Project Overview

This project demonstrates the complete workflow of building a text classification model to detect spam messages. It covers:

Data cleaning and preprocessing

Exploratory data analysis (EDA)

Text transformation using TF-IDF

Model training and evaluation using Naive Bayes, SVM, and Ensemble methods

📂 Dataset

The dataset used is the SMS Spam Collection Dataset, which contains 5,572 SMS messages labeled as ham or spam.

ham → Legitimate message

spam → Unwanted promotional or fraudulent message

⚙️ Technologies Used

Python

NumPy, Pandas – Data handling

Matplotlib, Seaborn – Data visualization

NLTK – Text preprocessing (tokenization, stemming, stopword removal)

Scikit-learn – Model building & evaluation

Pickle – Model saving

🔍 Data Preprocessing Steps

Converted all text to lowercase

Tokenized the text into words

Removed special characters, stopwords, and punctuation

Applied stemming using PorterStemmer

Transformed text into numerical features using TF-IDF Vectorizer

🤖 Models Used
Model	Description	Accuracy	Precision
Multinomial Naive Bayes	Best suited for text data	98.16%	99.17%
Support Vector Machine (SVM)	Kernel = Sigmoid	87%	50.68%
Extra Trees Classifier	Ensemble learning	–	–
Voting Classifier (Soft Voting)	Combines SVM, NB, and ExtraTrees	98.16%	99.17%

💾 Model Export


You can load them using:

import pickle

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

📊 Results

Accuracy: 98.16%

Precision: 99.17%

Excellent at detecting spam with minimal false positives.

🚀 How to Run

Clone this repository

git clone https://github.com/<your-username>/SMS-Spam-Prediction.git


Navigate to the project folder

cd SMS-Spam-Prediction


Install dependencies

pip install -r requirements.txt


Run the Jupyter Notebook or Python file to train/test the model.
