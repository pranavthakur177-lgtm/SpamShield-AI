import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page configuration
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Train model dynamically so no external .pkl files are required
@st.cache_resource
def load_trained_model():
    # Sample training corpus representing spam and ham patterns
    data = {
        'text': [
            'Congratulations you have won a free iPhone click here to claim your prize http://fake.com',
            'Urgent your bank account is suspended verify your password immediately',
            'Get 50% discount on casino games win cash now',
            'Hey are we meeting for lunch at the college canteen today?',
            'Please find attached the draft report for our mini project submission',
            'Reminder: Seminar scheduled for tomorrow at 10 AM in the auditorium',
            'Win a free vacation trip claim your reward right now'
        ],
        'label': [1, 1, 1, 0, 0, 0, 1]  # 1 = Spam, 0 = Ham
    }
    df = pd.DataFrame(data)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    
    model = MultinomialNB()
    model.fit(X, df['label'])
    
    return model, vectorizer

model, vectorizer = load_trained_model()

st.title("SpamShield AI")
st.markdown("### Intelligent Phishing & Spam Email Classification System")
st.write("Enter an email or text message below to check if it is spam or legitimate.")

# Input area
input_text = st.text_area("Enter message or email content here:")

if st.button("Classify Message", type="primary"):
    if input_text.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        st.markdown("---")
        st.subheader("Classification Result")
        
        # Transform input and predict
        input_vector = vectorizer.transform([input_text])
        prediction = model.predict(input_vector)[0]
        
        if prediction == 1:
            st.error("🚨 **Result: SPAM / Phishing Detected**")
            st.write("This message exhibits patterns typical of spam or fraudulent content.")
        else:
            st.success("✅ **Result: HAM (Legitimate Message)**")
            st.write("This message appears safe and non-spam.")