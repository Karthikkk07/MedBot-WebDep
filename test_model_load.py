from tensorflow.keras.models import load_model

try:
    model = load_model("chatbotmodel.h5")
    print("✅ Chatbot model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
