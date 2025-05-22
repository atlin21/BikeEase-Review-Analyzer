import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Loading the data
df = pd.read_csv("bike_rental_reviews.csv")
print(df.head())

df = df.dropna()

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# Text processing steps
# lower casing all the reviews
df['review_text'] = df['review_text'].str.lower()
# removing numbers
df['review_text'] = df['review_text'].str.replace(r'\d+', '', regex=True)
# removing punctuation
df['review_text'] = df['review_text'].str.replace(r'[^\w\s]', '', regex=True)
# removing whitespace
df['review_text'] = df['review_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
# removing stopwords
df['review_text'] = df['review_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
# these steps are to make it easier for the machine to read and understand the reviews, leading to better results. 

# Lemmatization, used to simplify words even further for the machine to read it.
lemmatizer = WordNetLemmatizer()

df['review_text'] = df['review_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(x)]))
df['token'] = df['review_text'].apply(nltk.word_tokenize) # tokenization

# Splitting train-test sets for models
X_train, X_test, y_train, y_test = train_test_split(df['review_text'], df['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# converting the simplified text to numeric so that the machine can read it

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_vec, y_train)
log_pred = log_model.predict(X_test_vec)
print("Logistic Regression:\n", classification_report(y_test, log_pred))
# Logistic Regression proved to be an amazing model for this type of prediction

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
print("Naive Bayes:\n", classification_report(y_test, nb_pred))
# Like the Logistic Regression, this model also proved to be very reliable for this prediction. 
# Both models very well could be suffering from overfitting.

# function to plot the confusion matricies for visualization of predictions making it easier to determine validity.
# reusability allows for less code being used, easier to read.
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=["positive", "neutral", "negative"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["positive", "neutral", "negative"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Labeling and encoding
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment'] = df['sentiment'].map(label_map)
# Giving the labels their associated numbers so the machine can classify each review in their proper category

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review_text'])
sequences = tokenizer.texts_to_sequences(df['review_text'])
padded = pad_sequences(sequences, padding='post', maxlen=100)

X_train, X_test, y_train, y_test = train_test_split(padded, df['sentiment'], test_size=0.2, random_state=42)

# LSTM model
model = Sequential([
    Embedding(10000, 64, input_length=100),
    LSTM(64),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))
# The LSTM model had about a 35% accuracy, not a very good mdoel for this data.

# encoding
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])  # Maps sentiment to 0,1,2

X_train, X_test, y_train, y_test = train_test_split(df['review_text'], df['label'], test_size=0.2, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128, return_tensors="tf")
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128, return_tensors="tf")

# BERT
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer_bert(list(X_train), truncation=True, padding=True, max_length=128, return_tensors="tf")
test_encodings = tokenizer_bert(list(X_test), truncation=True, padding=True, max_length=128, return_tensors="tf")

bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
bert_model.compile(optimizer=Adam(learning_rate=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

bert_model.fit(
    {
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask']
    },
    y_train,
    epochs=1
)

bert_logits = bert_model.predict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask']
}).logits
# Bert model didnt perform well either.

bert_pred = np.argmax(bert_logits, axis=1)
plot_confusion_matrix(y_test, log_pred, "Logistic Regression")
plot_confusion_matrix(y_test, nb_pred, "Naive Bayes")
plot_confusion_matrix(y_test, lstm_pred, "LSTM")
plot_confusion_matrix(y_test, bert_pred, "BERT")

# Overall, the simpler models performed better than the deep models. The data could've been too simple for the models which led to the poorer performance.

