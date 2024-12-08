import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from wordcloud import STOPWORDS
import nltk
from nltk.tokenize import word_tokenize
import string

# Load the model and vectorizer from pickle files
with open('resume_classifier_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    word_vectorizer = pickle.load(vectorizer_file)

# Function to clean the resume text
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)  # remove non-ASCII characters
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

# Streamlit user interface
st.title("Resume Categorization")

st.write("""
This is a resume classification tool. Upload your resume (PDF/Word) below to get the predicted category.
""")

# File uploader for resume
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    # Read the file content
    if uploaded_file.type == "text/plain":
        resume_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        # If PDF, extract text (you can use PyPDF2 or pdfplumber here for better extraction)
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # If DOCX, use python-docx to extract text
        from docx import Document
        doc = Document(uploaded_file)
        resume_text = ""
        for para in doc.paragraphs:
            resume_text += para.text

    # Clean the text
    cleaned_resume = cleanResume(resume_text)

    # Vectorize the text using the loaded vectorizer
    transformed_text = word_vectorizer.transform([cleaned_resume])

    # Make prediction using the trained model
    prediction = clf.predict(transformed_text)
    
    # Decode the predicted category
    category = {
    0: "Advocate",
    1: "Arts",
    2: "Automation Testing",
    3: "Blockchain",
    4: "Business Analyst",
    5: "Civil Engineer",
    6: "Data Science",
    7: "Database",
    8: "DevOps Engineer",
    9: "DotNet Developer",
    10: "ETL Developer",
    11: "Electrical Engineering",
    12: "HR",
    13: "Hadoop",
    14: "Health and Fitness",
    15: "Java Developer",
    16: "Mechanical Engineer",
    17: "Network Security Engineer",
    18: "Operation Manager",
    19: "PMO",
    20: "Python Developer",
    21: "SAP Developer",
    22: "Sales",
    23: "Testing",
    24: "Web Designing"
}
    # category = ["Category 1", "Category 2", "Category 3", "Category 4", "Category 5"]  # Adjust based on your categories
    predicted_category = category[prediction[0]]

    # Display the prediction
    st.write(f"Predicted Category: **{predicted_category}**")