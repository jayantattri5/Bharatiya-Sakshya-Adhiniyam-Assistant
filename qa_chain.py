import os
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

PROMPT_TEMPLATE_EN = """
You are a legal assistant specializing in the Bharatiya Sakshya Adhiniyam, 2023.
Use the context below to answer the question clearly and simply in English.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT_TEMPLATE_HI = """
आप 'भारतीय साक्ष्य अधिनियम, 2023' के विशेषज्ञ हैं।
नीचे दिए गए सन्दर्भ के आधार पर उत्तर दीजिए।
उत्तर हमेशा सरल और स्पष्ट हिंदी में हो।

सन्दर्भ:
{context}

प्रश्न:
{question}

उत्तर (केवल हिंदी में):
"""

INDEX_PATHS = {
    "en": "data/index_en",
    "hi": "data/index_hi"
}

EMBEDDING_MODELS = {
    "en": "sentence-transformers/all-MiniLM-L6-v2",
    "hi": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

PROMPTS = {
    "en": PROMPT_TEMPLATE_EN,
    "hi": PROMPT_TEMPLATE_HI
}

def get_qa_chain(lang="en", output_lang="en"):
    index_path = INDEX_PATHS[lang]
    embedding_model = EMBEDDING_MODELS[lang]
    prompt_template = PROMPTS[output_lang]

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma(persist_directory=index_path, embedding_function=embeddings)
    retriever = vectordb.as_retriever()

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa 