import sys
import streamlit as st
st.set_page_config(page_title="ATS Resume Analyzer", layout="centered")

import re
import time
import PyPDF2
import nltk
from nltk.corpus import stopwords
import pandas as pd
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt

# Predefined ATS keywords for job portal recommendations
predefined_ats_keywords = {
    "python", "data", "analysis", "machine", "learning", "sql", "excel",
    "communication", "teamwork", "project", "management", "problem-solving", "leadership"
}

# Sidebar navigation with emoticons
st.sidebar.title("🔍 ATS Resume Analyzer")
st.sidebar.write("Choose an option below:")
option = st.sidebar.radio("Select an option:", 
                          ["🏠 Home", "📊 ATS Score Check", "🚀 Job Recommendations", "✨ Resume Enhancer"])

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Instructions")
st.sidebar.write("1. For **Home**, view our landing page.")
st.sidebar.write("2. For **ATS Score Check**, upload your resume and enter a job description to extract ATS keywords.")
st.sidebar.write("3. For **Job Recommendations**, upload your resume for job portal suggestions.")
st.sidebar.write("4. For **Resume Enhancer**, upload your resume to get suggestions for improvements.")

# Main UI
if option == "🏠 Home":
    st.title("TOP HIRE 👨‍💻💡📝 ")
    st.title("Smart Resumes. Smart Applications. Smart Hires.🧠👨‍🎓 ")
else:
    title_map = {
        "📊 ATS Score Check": "📊 ATS Score Check",
        "🚀 Job Recommendations": "🚀 Job Recommendations",
        "✨ Resume Enhancer": "✨ Resume Enhancer"
    }
    st.title(title_map[option])
    st.write("Upload your **resume** file below and proceed with the selected option.")

    # Resume file uploader (for non-Home options)
    resume_file = st.file_uploader("Upload your resume file 📂", type=["txt", "pdf"], key="resume")
    if resume_file is not None:
        st.success(f"File '{resume_file.name}' uploaded successfully! 🎉")
        st.write("**File type:**", resume_file.type)

    # Function to extract text from the uploaded file
    def extract_text_from_file(uploaded_file):
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                return text
            else:
                st.error("Unsupported file type. Please upload a .txt or .pdf file.")
        return ""

    # Function to compute keyword frequencies
    def compute_keyword_frequencies(text, ats_keywords):
        text_words = re.findall(r'\w+', text.lower())
        frequencies = {keyword: text_words.count(keyword) for keyword in ats_keywords}
        return frequencies

    # ATS Score Check option
    if option == "📊 ATS Score Check":
        job_desc_text = st.text_area("Enter the job description text (for ATS keyword extraction) 📝")
        if st.button("Analyze ATS Score 🔍"):
            with st.spinner("Analyzing your resume... ⏳"):
                time.sleep(0.5)
                resume_text = extract_text_from_file(resume_file)
                if resume_text:
                    if job_desc_text:
                        # Extract ATS keywords from the job description text
                        ats_keywords = set(re.findall(r'\w+', job_desc_text.lower()))
                        # Find matched keywords in the resume text
                        matched_keywords = ats_keywords.intersection(set(re.findall(r'\w+', resume_text.lower())))
                        score = (len(matched_keywords) / len(ats_keywords)) * 100 if ats_keywords else 0
                        st.markdown("## ATS Score Analysis 📊")
                        st.markdown(f"**ATS Score:** {score:.2f}%")
                        st.markdown("**Matched ATS Keywords:**")
                        st.write(list(matched_keywords))
                        
                        # Compute keyword frequencies and plot using Matplotlib
                        frequencies = compute_keyword_frequencies(resume_text, ats_keywords)
                        frequencies_df = pd.DataFrame.from_dict(frequencies, orient='index', columns=['Count'])
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        frequencies_df.plot(kind='bar', ax=ax, legend=False)
                        ax.set_xlabel("Keywords")
                        ax.set_ylabel("Frequency")
                        ax.set_title("Keyword Frequency")
                        st.pyplot(fig)
                    else:
                        st.error("Please enter a valid job description text.")
                else:
                    st.error("Please upload a valid resume file.")

    # Job Recommendations option
    elif option == "🚀 Job Recommendations":
        if st.button("Get Job Recommendations 🚀"):
            with st.spinner("Analyzing your resume for job recommendations... ⏳"):
                time.sleep(0.5)
                resume_text = extract_text_from_file(resume_file)
                if resume_text:
                    matched_keywords = predefined_ats_keywords.intersection(set(re.findall(r'\w+', resume_text.lower())))
                    llm = ChatGroq(
                        temperature=0.6, 
                        groq_api_key='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', 
                        model_name="llama-3.2-1b-preview"
                    )
                    prompt = (
                        f"Based on the resume analysis which found the skills: {', '.join(matched_keywords)}, "
                        "please suggest some relevant job portals where this candidate might find suitable opportunities. "
                        "Include direct job apply links if possible."
                    )
                    groq_response = llm.invoke(prompt)
                    st.markdown("## Job Portal Recommendations 🚀")
                    st.write(groq_response.content)
                else:
                    st.error("Please upload a valid resume file.")

    # Resume Enhancer option
    elif option == "✨ Resume Enhancer":
        if st.button("Enhance Resume ✨"):
            with st.spinner("Enhancing your resume... ⏳"):
                time.sleep(0.5)
                resume_text = extract_text_from_file(resume_file)
                if resume_text:
                    llm = ChatGroq(
                        temperature=0.6,
                        groq_api_key='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',
                        model_name="llama-3.2-1b-preview"
                    )
                    prompt = (
                        f"Below is the resume provided:\n\n{resume_text}\n\n"
                        "Please suggest improvements and enhancements that can be made to this resume. "
                        "Provide constructive feedback and detailed suggestions for improvement."
                    )
                    groq_response = llm.invoke(prompt)
                    st.markdown("## Resume Enhancement Suggestions ✨")
                    st.write(groq_response.content)
                else:
                    st.error("Please upload a valid resume file.")

if __name__ == '__main__':
    pass
