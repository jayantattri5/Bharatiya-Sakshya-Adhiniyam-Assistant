import streamlit as st
from qa_chain import get_qa_chain
from langdetect import detect
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Bharatiya Sakshya Adhiniyam Assistant")
st.title("Bharatiya Sakshya Adhiniyam, 2023 – Legal Assistant")
st.write("Ask any question about the law. All answers are based strictly on the official document.")

# Language selection
col1, col2 = st.columns(2)
with col1:
    input_lang = st.selectbox("Select document language (for context): It means to say there are two vector embeddings one is the Hindi version of the document and the other is the English version of the document. [Choose English if your query is in Hinglish or English, Choose Hindi if your query is in Hindi]", ["en", "hi"], format_func=lambda x: "English" if x=="en" else "Hindi")
with col2:
    output_lang = st.selectbox("Select answer language:", ["en", "hi"], format_func=lambda x: "English" if x=="en" else "Hindi")

question = st.text_area("Your question (English, Hindi, or Hinglish):", height=80)

def detect_and_translate_question(q, context_lang):
    try:
        detected = detect(q)
    except Exception:
        detected = 'en'
    # Hinglish detection: if script is Latin but words are Hindi, treat as Hinglish
    # For simplicity, treat 'en' with many Hindi words as Hinglish
    # If context_lang is 'hi' and detected is 'en', translate to Hindi
    if context_lang == 'hi' and detected == 'en':
        # Try to translate to Hindi
        translator = GoogleTranslator(source='en', target='hi')
        translated = translator.translate(q)
        return translated
    elif context_lang == 'en' and detected == 'hi':
        # Translate to English
        translator = GoogleTranslator(source='hi', target='en')
        translated = translator.translate(q)
        return translated
    else:
        return q

def translate_answer(ans, output_lang):
    # If output_lang is 'hi', translate to Hindi; if 'en', to English
    if output_lang == 'hi':
        translator = GoogleTranslator(target='hi')
        return translator.translate(ans)
    elif output_lang == 'en':
        translator = GoogleTranslator(target='en')
        return translator.translate(ans)
    else:
        return ans

if st.button("Get Answer") and question.strip():
    with st.spinner("Thinking..."):
        # Detect and translate question if needed
        processed_question = detect_and_translate_question(question, input_lang)
        qa_chain = get_qa_chain(lang=input_lang, output_lang=output_lang)
        result = qa_chain({"query": processed_question})
        answer = result["result"]
        # Translate answer if needed
        final_answer = translate_answer(answer, output_lang)
        st.markdown(f"**Answer:**\n{final_answer}")
        if "source_documents" in result:
            with st.expander("Show source context"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}") 