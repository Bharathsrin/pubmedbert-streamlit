import streamlit as st
import csv
from datetime import datetime
import os

# Disable RTSDP (fixes meta tensor bug in PyTorch >=2.1)
os.environ["PYTORCH_USE_RTSDP"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from download_model import download_model
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    pipeline
)

# 📥 Download model if not already present
download_model()

# 🎨 Background styling
page_bg_img = '''
<style>
body {
    background-image: url('https://www.transparenttextures.com/patterns/cubes.png');
    background-size: cover;
    color: white;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# 📌 App title
st.title("🧠 Token Classification Web App")

# 🔧 Load model and tokenizer
model_path = "."
config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)

# 🔍 Set up pipeline (no manual .to(device) needed)
nlp_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy=None,
    device=-1
)

# 📄 Logging function
def log_interaction(user_input, predictions, log_file='interaction_logs.csv'):
    log_file_path = os.path.join(os.getcwd(), log_file)
    fieldnames = ['timestamp', 'user_input', 'predictions']

    try:
        file_exists = os.path.isfile(log_file_path)

        with open(log_file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            predictions_str = "; ".join([
                f"{entity['word']} ({entity.get('entity', 'N/A')}, {entity['score']:.4f})"
                for entity in predictions
            ])

            writer.writerow({
                'timestamp': timestamp,
                'user_input': user_input,
                'predictions': predictions_str
            })

        return log_file_path
    except Exception as e:
        st.error(f"Log saving error: {e}")
        return None

# 🧹 Clear log
if st.button("🗑️ Clear CSV Log"):
    log_file_path = os.path.join(os.getcwd(), 'interaction_logs.csv')
    with open(log_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'user_input', 'predictions'])
    st.success("✅ Interaction log cleared!")

# ✍️ Input
user_input = st.text_area("✍️ Enter your sentence:")

# 🔍 Prediction
if st.button("🔍 Analyze"):
    if user_input.strip():
        results = nlp_pipeline(user_input)

        # Clean predictions
        predictions = [{
            'word': entity.get('word', '[UNK]').replace("##", ""),
            'label': entity.get('entity', 'N/A'),
            'score': round(entity.get('score', 0.0), 4)
        } for entity in results]

        # Show tokens
        tokens = tokenizer.tokenize(user_input)
        st.markdown(f"**🧬 Tokens:** `{tokens}`")

        # Save logs
        log_filename = 'interaction_logs.csv'
        log_file_path = log_interaction(user_input, predictions, log_filename)

        if log_file_path:
            with open(log_file_path, 'r', encoding='utf-8') as file:
                log_file_content = file.read()

            st.download_button(
                label="📥 Download Interaction Log (CSV)",
                data=log_file_content,
                file_name=log_filename,
                mime="text/csv"
            )

        # Show predictions
        st.markdown("## 🔎 Prediction Results")
        for entity in predictions:
            html_output = f'''
            <div style='padding: 10px; background-color: rgba(0, 128, 255, 0.2); border-radius: 10px; margin-bottom: 8px;'>
                <strong>{entity['word']}</strong> — <code>{entity['label']}</code><br/>
                <progress value="{entity['score']}" max="1" style="width: 100%; height: 16px;"></progress> 
                <span>Score: {entity['score']:.2f}</span>
            </div>
            '''
            st.markdown(html_output, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text.")
