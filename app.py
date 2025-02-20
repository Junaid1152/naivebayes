import streamlit as st
import sklearn
import joblib
import pandas as pd

st.title('Spam Detector Interface')

df = pd.read_csv("spam.csv", encoding = 'ISO-8859-1')
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.rename(columns={'v1':'Target','v2':'Features'},inplace=True)
from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
df['Target'] = Encoder.fit_transform(df['Target'])
X = df['Features']
y = df['Target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.01,random_state=10)
from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer()
X_train_vec = Vectorizer.fit_transform(X_train)
X_test_vec = Vectorizer.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_vec,y_train)

def prediction(email):
    vectorized_email = Vectorizer.transform([email]) # Vectorizing The Email Content
    prediction = model.predict(vectorized_email) # Making Predictions From The Model
    return prediction[0]
user_input = st.text_area('Enter The Email To Classify')
if st.button('Predict'):
    result = prediction(user_input)
    if result == 1:
        st.error('This Email Is Classified As Spam.')
    else:
        st.success('This Email Is Classified As Ham.')
