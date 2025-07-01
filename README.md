##This project aims to classify incoming emails as spam or ham (not spam) using machine learning techniques. It was developed in a Jupyter Notebook environment using Python.

🔍 Project Objective
The main goal is to build a model that automatically detects whether an email is spam based on its content. This can help filter out unwanted emails more efficiently.

🧰 Libraries Used
-pandas
-numpy
-matplotlib
-seaborn
-sklearn (TfidfVectorizer, train_test_split, models, metrics)
-nltk (for stopwords and stemming)

📁 Dataset
The dataset used is spam.csv, which includes:

-v1: The label (spam or ham)
-v2: The message content
After loading, the dataset is cleaned by removing unnecessary columns and applying text preprocessing.

⚙️ Project Workflow
Data Loading & Cleaning:

-Load the CSV file and drop irrelevant columns.
-Convert categorical labels (spam, ham) to binary values.

Text Preprocessing:

-Convert text to lowercase.
-Remove punctuation and stopwords.
-Apply stemming.
-Vectorize using TF-IDF.

Model Training:

-Split the dataset into training and testing sets.
-Train and evaluate different models:
-Naive Bayes (MultinomialNB)
-Logistic Regression
-Support Vector Machine (SVM)

Evaluation:

-Use metrics like accuracy, precision, recall, and F1-score to compare models.

📊 Results
Models like SVM and Logistic Regression performed well. Preprocessing techniques (especially TF-IDF and stopword removal) played a crucial role in improving classification performance.

📌 Notes
You may need to download NLTK resources (e.g., stopwords and punkt):
Make sure the spam.csv dataset is placed in the same directory as the notebook, or update the file path accordingly.
