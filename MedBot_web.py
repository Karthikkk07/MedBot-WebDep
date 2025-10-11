import streamlit as st
import json
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# --- Chatbot Setup ---
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbotmodel.h5')

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
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    if len(ints) == 0:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"
    tag = ints[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure I understand that."

def chatbot_response(text):
    ints = predict_class(text)
    res = get_response(ints, intents)
    return res

# --- Heart Disease Prediction Model ---
@st.cache_data
def train_heart_model():
    df = pd.read_csv("heart-disease.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    return model

heart_model = train_heart_model()

# --- Streamlit UI ---
st.set_page_config(page_title="MedBot - AI Health Assistant", page_icon="🩺", layout="centered")

st.title("🩺 MedBot - AI Health Assistant")
st.markdown("#### Your personal healthcare companion powered by AI and Machine Learning")

tab1, tab2 = st.tabs(["💬 Chatbot", "❤️ Heart Disease Predictor"])

# --- Chatbot Tab ---
with tab1:
    st.subheader("Chat with MedBot 🤖")
    user_input = st.text_input("You:", "")
    if st.button("Ask"):
        if user_input.strip() != "":
            response = chatbot_response(user_input)
            st.text_area("MedBot:", value=response, height=120)
        else:
            st.warning("Please enter a message.")

# --- Heart Disease Prediction Tab ---
with tab2:
    st.subheader("Heart Disease Risk Prediction ❤️")
    st.markdown("Enter your health parameters:")

    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex", [0, 1], help="0 = female, 1 = male")
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (0=normal,1=fixed,2=reversible)", [0, 1, 2])

    if st.button("Predict"):
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                              exang, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(features)
        if prediction[0] == 1:
            st.error("⚠️ You may be at risk of heart disease. Please consult a doctor.")
        else:
            st.success("✅ You are likely healthy. Keep maintaining a good lifestyle!")
