# Text Classification using NLP
This project focuses on performing text classification using Natural Language Processing (NLP) techniques. The objective is to classify text data into different categories based on its content.
![image](https://github.com/Aftabbs/Text-Classification-Using-NLP/assets/112916888/7d9dd480-c870-4c58-b80a-bd4291b8d790)

# Dataset
The dataset used for this project is stored in sms.tsv. It contains text messages along with their corresponding labels.

# Data Preprocessing
The text data is preprocessed to convert it into numerical format, which can be fed into machine learning models. Various preprocessing steps include:

* Removing special characters and punctuation.
* Converting all text to lowercase.
* Removing stop words.
* Lemmatization and stemming to reduce words to their base forms.
* Vectorization

Two different vectorization techniques are used to represent text data numerically:

* CountVectorizer: Converts text into a matrix of token counts.
* TFIDFVectorizer: Transforms text into a matrix of TF-IDF features.

# Model Building and Evaluation
Several machine learning models are used to classify the text data, including:

* Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
The models are evaluated based on various metrics such as accuracy, precision, recall, and F1-score.

# Comparing Models
The performance of different models is compared to identify the best performing model for the text classification task.

# Fine-tuning Vectorizer
The vectorizer is fine-tuned by adjusting its parameters to optimize the model's performance.

# Word Cloud
A word cloud is generated to visualize the most frequent words in the text data.

# Sentiment Calculation
Sentiment analysis is performed on the text data to classify the sentiment of each message as positive, negative, or neutral.

# Dependencies
The following libraries are required to run the code:
* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn
* nltk
* wordcloud
Make sure to install these dependencies before running the notebook.

# Contact
If you have any questions or feedback, feel free to contact me at [your_email@example.com].

Happy text classification with NLP!






