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

# Download required NLTK data (using wordpunct_tokenize avoids the 'punkt_tab' issue)
nltk.download('stopwords')

# Predefined ATS keywords for job portal recommendations and skill extraction
predefined_ats_keywords = {
    "python", "data", "analysis", "machine", "learning", "sql", "excel",
    "communication", "teamwork", "project", "management", "problem-solving", "leadership"
}

# Define navigation options (new option added for Cold Email Generator)
options = [
    "🏠 Home",
    "👤 Personal ATS Score Checker",
    "📊 ATS Score Checker(Corporate)",
    "🚀 Job Recommendations",
    "✨ Resume Enhancer",
    "📧 Cold Email Generator"
]

# Update navigation if a redirect is requested
if 'redirect' in st.session_state:
    st.session_state.nav_option = st.session_state.redirect
    del st.session_state.redirect

# Initialize nav_option in session state if not already set
if 'nav_option' not in st.session_state:
    st.session_state.nav_option = options[0]

# Sidebar navigation using session state
option = st.sidebar.radio("Select an option:", options, key="nav_option")

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Instructions")
st.sidebar.write("1. For **Home**, view our landing page.")
st.sidebar.write("2. For **Personal ATS Score Checker**, upload your resume and enter a job description to check your personal ATS score. Use the navigation buttons to jump to Resume Enhancer or Job Recommendations.")
st.sidebar.write("3. For **ATS Score Checker**, upload your resume(s) for bulk analysis.")
st.sidebar.write("4. For **Job Recommendations**, upload your resume to get job portal suggestions.")
st.sidebar.write("5. For **Resume Enhancer**, upload your resume to receive improvement suggestions.")
st.sidebar.write("6. For **Cold Email Generator**, upload your resume to generate a professional cold email based solely on the skills mentioned in your resume.")

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

# Function to compute keyword frequencies using NLTK's wordpunct_tokenize
def compute_keyword_frequencies(text, ats_keywords):
    tokens = nltk.tokenize.wordpunct_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]  # filter out non-alphabetical tokens
    frequencies = {keyword: tokens.count(keyword) for keyword in ats_keywords}
    return frequencies

# Main UI
if option == "🏠 Home":
    st.title(" CareerFit 👨‍💻💡📝")
    st.title("Smart Resumes. Smart Applications. Smart Hires.🧠👨‍🎓")

# Personal ATS Score Checker page
elif option == "👤 Personal ATS Score Checker":
    st.title("Personal ATS Score Checker")
    personal_resume = st.file_uploader("Upload your resume file 📂", type=["txt", "pdf"], key="personal_resume")
    personal_job_desc = st.text_area("Enter your job description (for ATS analysis) 📝", key="personal_job_desc")
    if st.button("Check My ATS Score"):
        if personal_resume and personal_job_desc:
            with st.spinner("Analyzing your resume... ⏳"):
                time.sleep(0.5)
                resume_text = extract_text_from_file(personal_resume)
                if resume_text:
                    ats_keywords = set(re.findall(r'\w+', personal_job_desc.lower()))
                    matched_keywords = ats_keywords.intersection(set(re.findall(r'\w+', resume_text.lower())))
                    score = (len(matched_keywords) / len(ats_keywords)) * 100 if ats_keywords else 0
                    st.markdown("## Personal ATS Score Analysis 📊")
                    st.markdown(f"**Your ATS Score:** {score:.2f}%")
                    st.markdown("**Matched ATS Keywords:**")
                    st.write(list(matched_keywords))
                    
                    # Compute and display keyword frequencies
                    frequencies = compute_keyword_frequencies(resume_text, ats_keywords)
                    frequencies_df = pd.DataFrame.from_dict(frequencies, orient='index', columns=['Count'])
                    fig, ax = plt.subplots(figsize=(8, 4))
                    frequencies_df.plot(kind='bar', ax=ax, legend=False)
                    ax.set_xlabel("Keywords")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Keyword Frequency")
                    st.pyplot(fig)
                    
                    # Navigation buttons to other sections
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Go to Resume Enhancer", key="nav_to_enhance"):
                            st.session_state.redirect = "✨ Resume Enhancer"
                            st.experimental_rerun()
                    with col2:
                        if st.button("Go to Job Recommendations", key="nav_to_recommend"):
                            st.session_state.redirect = "🚀 Job Recommendations"
                            st.experimental_rerun()
                else:
                    st.error("Could not extract text from the resume file.")
        else:
            st.error("Please upload a resume file and enter a job description.")

# ATS Score Check page (for multiple resumes)
elif option == "📊 ATS Score Checker(Corporate)":
    st.title("📊 ATS Score Checker(Corporate)")
    st.write("Upload your resume file(s) below and enter a job description for bulk ATS keyword extraction.")
    resume_files = st.file_uploader("Upload your resume files (up to 3) 📂", 
                                    type=["txt", "pdf"], key="resume_bulk", accept_multiple_files=True)
    if resume_files:
        for resume_file in resume_files:
            st.success(f"File '{resume_file.name}' uploaded successfully! 🎉")
            st.write("**File type:**", resume_file.type)
    job_desc_text = st.text_area("Enter the job description text (for ATS keyword extraction) 📝")
    if st.button("Analyze ATS Score 🔍"):
        if not resume_files:
            st.error("Please upload at least one resume file.")
        elif not job_desc_text:
            st.error("Please enter a valid job description text.")
        else:
            with st.spinner("Analyzing your resumes... ⏳"):
                time.sleep(0.5)
                ats_keywords = set(re.findall(r'\w+', job_desc_text.lower()))
                st.markdown("## ATS Score Analysis 📊")
                for resume_file in resume_files:
                    resume_text = extract_text_from_file(resume_file)
                    if resume_text:
                        matched_keywords = ats_keywords.intersection(set(re.findall(r'\w+', resume_text.lower())))
                        score = (len(matched_keywords) / len(ats_keywords)) * 100 if ats_keywords else 0
                        st.markdown(f"### Resume: {resume_file.name}")
                        st.markdown(f"**ATS Score:** {score:.2f}%")
                        st.markdown("**Matched ATS Keywords:**")
                        st.write(list(matched_keywords))
                        
                        frequencies = compute_keyword_frequencies(resume_text, ats_keywords)
                        frequencies_df = pd.DataFrame.from_dict(frequencies, orient='index', columns=['Count'])
                        fig, ax = plt.subplots(figsize=(8, 4))
                        frequencies_df.plot(kind='bar', ax=ax, legend=False)
                        ax.set_xlabel("Keywords")
                        ax.set_ylabel("Frequency")
                        ax.set_title(f"Keyword Frequency for {resume_file.name}")
                        st.pyplot(fig)
                    else:
                        st.error(f"Could not extract text from {resume_file.name}.")

# Job Recommendations page
elif option == "🚀 Job Recommendations":
    st.title("🚀 Job Recommendations")
    st.write("Upload your resume file below to get job portal suggestions.")
    resume_file = st.file_uploader("Upload your resume file 📂", type=["txt", "pdf"], key="resume_recommend")
    if resume_file is not None:
        st.success(f"File '{resume_file.name}' uploaded successfully! 🎉")
        st.write("**File type:**", resume_file.type)
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

# Resume Enhancer page
elif option == "✨ Resume Enhancer":
    st.title("✨ Resume Enhancer")
    st.write("Upload your resume file below to get suggestions for improvements.")
    resume_file = st.file_uploader("Upload your resume file 📂", type=["txt", "pdf"], key="resume_enhance")
    if resume_file is not None:
        st.success(f"File '{resume_file.name}' uploaded successfully! 🎉")
        st.write("**File type:**", resume_file.type)
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

# Cold Email Generator page (updated: based solely on resume skills)
elif option == "📧 Cold Email Generator":
    st.title("📧 Cold Email Generator")
    st.write("Upload your resume to generate a professional cold email based on the skills mentioned in your resume.")
    resume_file = st.file_uploader("Upload your resume file 📂", type=["txt", "pdf"], key="resume_coldemail")
    if st.button("Generate Cold Email"):
        with st.spinner("Generating your cold email... ⏳"):
            time.sleep(0.5)
            resume_text = extract_text_from_file(resume_file)
            if resume_text:
                # Extract skills based on predefined ATS keywords
                matched_skills = predefined_ats_keywords.intersection(set(re.findall(r'\w+', resume_text.lower())))
                llm = ChatGroq(
                    temperature=0.6,
                    groq_api_key='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',
                    model_name="llama-3.2-1b-preview"
                )
                prompt = (
                    f"Based on the resume content provided below:\n\n{resume_text}\n\n"
                    f"The identified skills from the resume are: {', '.join(matched_skills)}.\n\n"
                    "Please generate a professional cold email that highlights these skills, introduces the candidate, "
                    "and expresses interest in job opportunities. The email should be tailored to hiring companies."
                )
                groq_response = llm.invoke(prompt)
                st.markdown("## Generated Cold Email")
                st.write(groq_response.content)
            else:
                st.error("Please upload a valid resume file.")

if __name__ == '__main__':
    pass
