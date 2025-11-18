# Email & SMS Multi-Class Spam Detection using NLP and Machine Learning

A unified machine learning system designed to classify both email and SMS messages into four categories:

* Email Ham
* Email Spam
* SMS Ham
* SMS Spam

The system uses NLP techniques, TF-IDF vectorization, and security-based metadata features to enhance spam detection accuracy and enable real-time classification with model confidence scores.

---

## 1. Project Objectives

* Develop a single, multi-class system for detecting spam in emails and SMS.
* Apply NLP preprocessing, TF-IDF, and security-oriented metadata extraction.
* Train multiple machine learning models and compare their performance.
* Support real-time message classification with probability scores.

---

## 2. Dataset Details

The dataset consists of the following columns:

| Column | Description                    |
| ------ | ------------------------------ |
| text   | Message content (Email or SMS) |
| label  | Spam or Ham                    |
| type   | Message type: Email or SMS     |

### Data Cleaning Performed

* Removed missing or invalid entries
* Standardized labels
* Created unified 4-class output:

  * `email_ham`
  * `email_spam`
  * `sms_ham`
  * `sms_spam`

---

## 3. NLP Processing Pipeline

* Text cleaning
* Tokenization
* Stopword removal
* Lemmatization
* TF-IDF vectorization
* Security Feature Extraction:

  * URL count
  * Suspicious domains
  * Phishing-related keywords
  * Presence of all-caps words
  * Exclamation marks and special characters

---

## 4. Machine Learning Models

The following models were trained and evaluated:

* Multinomial Naive Bayes
* Logistic Regression
* Random Forest Classifier
* K-Nearest Neighbors (KNN)

Each model was tested on the multi-class dataset using standard evaluation metrics.

---

## 5. Results and Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1-score

### Key Insights

* Logistic Regression and Random Forest showed strong multi-class performance.
* Naive Bayes performed consistently well for text-dominated datasets.
* KNN performance varied depending on dataset size and feature dimensionality.

### Visualizations Included

* Classification reports
* Confusion matrices
* Model comparison charts
* Probability/Confidence score plots

---

## 6. Conclusion

A unified multi-class spam detection system effectively classifies both emails and SMS messages using a hybrid of NLP and security-based features.
The system demonstrates strong performance across multiple models and is suitable for real-time spam detection use cases.

---

## 7. Future Enhancements

* Integration of deep learning models (LSTM, BERT)
* Multilingual detection capabilities
* Deployment as an API or mobile/desktop application
* Continuous model learning using streaming data
* Expansion with real-world spam datasets

---

## 8. References

* [Scikit-learn Documentation](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html scikit-learn)
* NLTK NLP Toolkit
*  [Project Dataset](Email_SMS_Data.csv)
* https://etasr.com/index.php/ETASR/article/view/7631
* Python and Streamlit official documentation

