import streamlit as st

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# Функция для предсказания корректности кода
def predict_code_correctness(code):
    # Токенизация и предсказание
    inputs = tokenizer(code, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits).item()
    return "correct" if predicted_class_id == 1 else "incorrect"

# Интерфейс Streamlit
st.title("JavaScript Code Correctness Checker")

# Информация об авторстве
st.markdown("""
    **Автор:** Ридлиgit -v  
    **Контакт:** ваша.почта@example.com  
    **Описание:** Это приложение использует обученную модель CodeBERT для проверки корректности JavaScript кода.
""")

code = st.text_area("Введите JavaScript код для проверки", height=200)

if st.button("Проверить"):
    result = predict_code_correctness(code)
    st.write(f"Результат: {result}")
