import os
import openai
import tiktoken
import json
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from typing import Optional
from langchain.document_loaders import PyPDFLoader
import gradio as gr
from langchain.chains import create_structured_output_runnable
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import DirectoryLoader
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

def upload_pdf_and_get_documents():
    pdf_file = st.file_uploader("Upload document", type=["pdf"])
    if pdf_file is not None:
        # Save uploaded file
        save_path = os.path.join('./report', pdf_file.name)
        with open(save_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # Load documents
        pdf_loader = DirectoryLoader('./report', glob='**/*.pdf')
        return pdf_loader.load()
    return []

def detect_language(documents):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    prompt_template = """
    Detect the language of the given text:
    "{context}"
    give output as a JSON 
    for example if the detected language is English
    it should output a dictionary like this:
    Language: English
    """
    prompt = PromptTemplate.from_template(prompt_template)
    lang_detector_chain = create_stuff_documents_chain(llm, prompt)
    return lang_detector_chain.invoke({"context": documents})

def summarize_documents(documents):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    prompt_template = """
    Write a concise summary of the following:
    "{context}"
    give output as a JSON 
    it should output a dictionary like this:
    Concise_summary: summary of the input text"""
    prompt = PromptTemplate.from_template(prompt_template)
    summarizer_chain = create_stuff_documents_chain(llm, prompt)
    summarizer = summarizer_chain.invoke({"context": documents})
    # Parse the string dictionary as JSON
    json_dict = json.loads(summarizer)
    # Extract the value associated with the key "Concise_summary"
    concise_summary = json_dict["Concise_summary"]
    return concise_summary


def translate_text(text):
    class Translator(BaseModel):
        '''Translate the text.'''
        translated_text: str = Field(..., description="Translate the text into French if detected language is English or vice-versa.")

    llm_translation = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm = create_structured_output_runnable(Translator, llm_translation, mode="openai-functions")
    translation_result = structured_llm.invoke(text)
    return translation_result.translated_text

# Title and description
st.title("PDF Summarizer: Get the gist out of your data")
st.subheader("Upload your PDF file and get a concise summary!")
st.divider()

# File uploader and summary button
pdf_documents = upload_pdf_and_get_documents()
if st.button("Summary") and pdf_documents:
    lang_detection_result = detect_language(pdf_documents)
    summarizer_result = summarize_documents(pdf_documents)
    translated_text = translate_text(summarizer_result)
    
    # Display summaries
    st.header('Summaries in both French and English Language')
    col1, col2 = st.columns([5,5])

    with col1:
        st.header("English Summary")
        st.write(summarizer_result)

    with col2:
        st.header("French Summary")
        st.write(translated_text)

reset = st.sidebar.button('Reset all')
if reset:
   st.session_state.clear()
