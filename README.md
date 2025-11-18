# A4_G3_AIML
Email and SMS Spam vs Ham Detection using NLP tectniques and ML Algorithms
Multi-Class Email & SMS Spam Detection System

A machine-learning based system that classifies messages into Email Ham, Email Spam, SMS Ham, SMS Spam using NLP and security-based features.

1. Project Title & Objectives
Project Title:

Email & SMS Multi-Class Spam Detection Using NLP and Machine Learning

Objectives:

Build a unified spam detection system for both email and SMS messages.

Use NLP preprocessing, TF-IDF, and security-based metadata to improve accuracy.

Train multiple ML models and compare their performance.

Allow real-time message classification with confidence scores.

2. Dataset Details

The dataset must contain the following columns:

Column	Description
text	Message content (email or SMS)
label	spam or ham
type	email or sms

Data cleaning steps:

Removed missing values

Standardized labels

Combined labels into 4 classes:

email_ham, email_spam, sms_ham, sms_spam

3. Algorithms & Models Used
NLP Processing

Text cleaning

Stopword removal

Lemmatization

TF-IDF vectorization

Security feature extraction (URLs, suspicious domains, phishing keywords, all-caps, exclamations)

Machine Learning Models

Multinomial Naive Bayes

Logistic Regression

Random Forest Classifier

K-Nearest Neighbors (KNN)

Each model is trained and evaluated on the multi-class dataset.

4. Results

Models evaluated on accuracy, precision, recall, and F1-score.

Typical findings:

Logistic Regression and Random Forest perform well for multi-class classification.

Naive Bayes performs consistently good on text data.

KNN performs moderately depending on dataset size.

Performance visualization includes:

Classification reports

Confusion matrices

Confidence probability charts

5. Conclusion

A unified multi-class system can successfully classify both email and SMS messages into spam or ham.

Combining NLP with security-based features significantly improves detection accuracy.

The system can be used for real-time spam detection across multiple communication channels.

6. Future Scope

Integrate deep learning models (LSTM, BERT).

Add language detection and multilingual support.

Deploy as an API or mobile app.

Implement continuous learning with streaming data.

Expand dataset with real-world spam samples.

7. References

Scikit-learn documentation

NLTK NLP library documentation

Research papers on phishing and spam detection

Public spam datasets (SpamAssassin, SMS Spam Collection)

Python & Streamlit official documentation
