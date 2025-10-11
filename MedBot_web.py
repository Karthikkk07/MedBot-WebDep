import streamlit as st
import json
import random
import nltk
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# ----------------- Fix NLTK Errors -----------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# ----------------- Chatbot Setup -----------------
lemmatizer = WordNetLemmatizer()

model = load_model('chatbotmodel.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# ----------------- Helper Functions -----------------
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_vec = bow(sentence, words)
    res = model.predict(np.array([bow_vec]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm sorry, I didn't quite understand that."

def chatbot_response(text):
    ints = predict_class(text)
    res = get_response(ints, intents)
    return res

# ----------------- Heart Disease Prediction -----------------
def train_heart_model():
    df = pd.read_csv('heart-disease.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    joblib.dump((model, scaler), 'heart_model.pkl')

def load_heart_model():
    try:
        model, scaler = joblib.load('heart_model.pkl')
    except:
        train_heart_model()
        model, scaler = joblib.load('heart_model.pkl')
    return model, scaler

def predict_heart_disease(inputs):
    model, scaler = load_heart_model()
    inputs_scaled = scaler.transform([inputs])
    prediction = model.predict(inputs_scaled)
    return "High risk of heart disease 💔" if prediction[0] == 1 else "Low risk of heart disease 💖"

# ----------------- Streamlit Web UI -----------------
st.set_page_config(page_title="MedBot Web", page_icon="💊", layout="centered")

st.title("💊 MedBot – AI Health Assistant")
st.write("Ask me health questions or predict heart disease risk!")

menu = st.sidebar.radio("Choose an option:", ["Chatbot", "Heart Disease Predictor"])

# Chatbot Interface
if menu == "Chatbot":
    st.header("💬 Health Chatbot")
    user_input = st.text_input("You:", placeholder="Type your question here...")
    if st.button("Ask"):
        if user_input.strip() != "":
            response = chatbot_response(user_input)
            st.success(response)
        else:
            st.warning("Please enter a question.")

# Heart Disease Prediction Interface
elif menu == "Heart Disease Predictor":
    st.header("🫀 Heart Disease Prediction")

    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 70, 220)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

    if st.button("Predict"):
        inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        result = predict_heart_disease(inputs)
        st.success(result)

st.sidebar.markdown("---")
st.sidebar.info("Developed by Karthik 💻 | Powered by Streamlit & TensorFlow")
