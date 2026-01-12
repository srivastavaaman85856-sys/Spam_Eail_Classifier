import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Streamlit UI
st.title("📧 Spam Email Classifier")
st.write("Enter an email message to check whether it is Spam or Not Spam.")

user_input = st.text_area("Enter email text here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.error("🚫 This email is SPAM")
        else:
            st.success("✅ This email is NOT SPAM")
