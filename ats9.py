from dotenv import load_dotenv
import base64
import streamlit as st
import os
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document
import google.generativeai as genai
import plotly.express as px
import requests
from io import StringIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlencode
import pandas as pd
from fpdf import FPDF
import re

# Load API key from environment variables
load_dotenv()
genai.configure(api_key="AIzaSyAO3d8aviiNDoxKXYgBS2ywp_9ko3o8H28")

# Initialize sentence transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate response from Gemini AI model
def get_gemini_response(input_text, pdf_content, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([input_text, pdf_content, prompt])
    return response.text

# Function to process the uploaded document and generate FAISS embeddings
def process_document_with_faiss(uploaded_file, file_type):
    if uploaded_file is not None:
        text = ""
        if file_type == "pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = " ".join([page.extract_text() for page in pdf_reader.pages])
        elif file_type == "docx":
            doc = Document(uploaded_file)
            text = " ".join([para.text for para in doc.paragraphs])
        elif file_type == "txt":
            text = uploaded_file.getvalue().decode("utf-8")

        # Chunk the text into manageable sizes
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        
        # Generate embeddings for each chunk
        embeddings = embedder.encode(chunks, convert_to_tensor=True)
        
        # Create a FAISS index for the embeddings
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.cpu().numpy())
        
        return chunks, index, embeddings
    else:
        raise FileNotFoundError("No file uploaded")

# Function to fetch LinkedIn profile data (simplified version)
def get_linkedin_data(linkedin_url):
    # Example URL: "https://www.linkedin.com/in/username"
    # Note: In a real scenario, scraping or using LinkedIn's API would be required.
    try:
        response = requests.get(linkedin_url)
        if response.status_code == 200:
            return response.text  # Return raw HTML (in a real scenario, we would parse it)
        else:
            return "Error fetching LinkedIn profile data"
    except Exception as e:
        return str(e)

# Function to get LinkedIn job recommendations based on the first line of the job description
def get_linkedin_job_recommendations(job_description):
    try:
        # Extract the first line of the job description
        first_line = job_description.split('\n')[0].strip()
        location = "India"

        # Construct query parameters
        query_params = {
            "keywords": first_line,
            "location": location,
        }

        # Encode parameters for use in the URL
        encoded_params = urlencode(query_params)
        linkedin_jobs_url = f"https://www.linkedin.com/jobs/search?{encoded_params}"
        return linkedin_jobs_url
    except Exception as e:
        return str(e)

# Function to generate interview preparation questions
def generate_interview_questions(job_description):
    try:
        prompt = f"Generate 5 short one lined technical interview questions only for the specified job role, categorized into fresher, intermediate, and expert levels, with 5 questions for each level.\n{job_description}"
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        return str(e)

# Streamlit App
st.set_page_config(page_title="Applicant Tracking System", page_icon=":Briefcase:", layout="wide", menu_items=None)

# Custom CSS for enhanced design
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
  /* Sidebar styles */
  .sidebar-container {
    background-color: #f8f8f8;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  }
  .sidebar-title {
    font-family: 'Arial', sans-serif;
    color: #333333;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 20px;
  }
  .sidebar-text {
    font-family: 'Arial', sans-serif;
    color: #666666;
    font-size: 16px;
  }
  /* Footer styles */
  .footer {
    background-color: #333333;
    color: #ffffff;
    padding: 10px 0;
    width: 100%;
    text-align: center;
    font-size: 14px;
    position: fixed;
    bottom: 0;
    left: 0;
    z-index: 1;
  }
  /* Main content styles */
  .main-content {
    margin-left: 300px; /* Sidebar width */
    padding-bottom: 60px; /* Footer height */
  }
  
</style>
""", unsafe_allow_html=True)

# Login Page
st.title("Applicant Tracking System")
user_type = st.radio("Select User Type:", ("Employee", "Recruiter"))
# Main content area
with st.container():
    col2 = st.columns([3])[0]
    
    st.markdown("""
    ## Application Tracking System:
    An Applicant Tracking System (ATS) is a software application used by organizations to manage the recruitment process.   
    ## How Does an ATS Work?
    An ATS works by automating and centralizing the recruitment process, making it easier for HR professionals to handle multiple applicants, track application statuses and manage candidate communications.
    """)
    
st.image("C:/Users/mahat/Desktop/ATS/Applicant-Tracking-System/atsimage.jpg",width=800)

# Sidebar
if user_type == "Employee":
    with st.sidebar:
        st.title("AI-Optimized Applicant Tracking System")
        st.text("Precision Resume Screening & Role Fit Analysis")
        input_text = st.text_area("Job Description: ", key="input")
        uploaded_file = st.file_uploader("Upload Your Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
        linkedin_url = st.text_input("LinkedIn Profile URL (Optional):")
        with st.form(key='my_form'):
            submit1 = st.form_submit_button("See Results")
            submit3 = st.form_submit_button("Make A Match")


    st.markdown("""  
        ---                         
        > ## Optimize your resume in minutes
        > ###### UPLOAD RESUME --> ATS SCAN --> VIEW RESULTS                
        ---
        ## What our ATS resume scanner checks

        - [x] **Customization**
        - [x] **Spelling and Grammar**
        - [x] **Summary Statement**
        - [x] **Missing Information**
        - [x] **Word Choice**
        - [x] **Formatting**
        - [x] **Optimal Length**
        - [x] **Contact Information**
        - [x] **Comprehensiveness**


        """)

    if submit1 or submit3:
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1]
            chunks, faiss_index, resume_embeddings = process_document_with_faiss(uploaded_file, file_type)
                
            # Generate embeddings for the job description
            job_desc_embedding = embedder.encode([input_text], convert_to_tensor=True).cpu().numpy()
                
            # Calculate cosine similarity between job description and each resume chunk
            cosine_similarities = cosine_similarity(job_desc_embedding, resume_embeddings.cpu().numpy())
            
            # Calculate the average cosine similarity to determine the match percentage
            average_similarity = np.mean(cosine_similarities)
            match_percentage = average_similarity * 100
            non_match_percentage = 100 - match_percentage

            # Prepare the content for analysis
            pdf_content = " ".join(chunks)
            input_prompt = """
                You are an experienced Technical Human Resource Manager. Analyze the provided resume content and job description.
                Goal:
            We want our Applicant Tracking System (ATS) software to automatically analyze resumes and evaluate how well they match a specific job description. The ATS will provide a detailed report based on the analysis.

            Inputs for the ATS:

            Resume: 
            Job Description: 
            Outputs from the ATS:The ATS should provide a comprehensive neatly formatted report with the following information:(use bullets wherever possible)

            1. Searchability
            Overall ATS Score: A score (0-100) showing how well the resume is optimized for ATS systems.
            Keyword Matching:
            Matched Keywords: A list of job description keywords found in the resume.
            Missing Keywords: A list of keywords from the job description that are absent in the resume.
            Keyword Frequency: The frequency of each keyword in the resume.
            Can the ATS Read the Resume?:
            A score (0-100) indicating how easily the ATS can extract information from the resume based on its structure.
            List any potential formatting issues (e.g., images, unusual fonts) that could hinder ATS processing.

            2. Skills
            Technical Skills: A list of technical skills mentioned in the resume (e.g., programming languages, software).
            Soft Skills: A list of soft skills mentioned in the resume (e.g., teamwork, communication).
            Missing Skills: A list of skills from the job description that are not found in the resume.

            3. Formatting
            Formatting Score:A score (0-100) indicating how well the resume is formatted for both ATS systems and human readers.
            Suggestions for Improvement: If the formatting could be better, provide advice on optimizing the layout for clarity and ATS compatibility.
            Customization Score: A score (0-100) on how well the resume is tailored to the job description.
            Suggestions: How to better customize the resume to highlight the most relevant experiences or skills.

            4. Recruiter Tips (in bullets)


            """ if submit1 else """

        
                Analyze the resume and calculate the percentage match with the job description.
                Provide:
                
                
                1.Customization: A score (0-100) on how well the resume matches the job description in terms of relevant skills and experience.
                Suggestions: How to further align the resume with the job description.

                2.Spelling and Grammar: A score (0-100) based on spelling and grammatical errors found.
                List of actual errors found and suggestions for correction.

                3.Summary:A score (0-100) on the effectiveness of the resume summary.
                Suggestions: Recommendations to improve the summary if it’s missing or weak.
        
                4.Achievements: A score evaluating whether the resume includes measurable results (e.g., performance metrics, key achievements).
                Suggestions: How to include quantifiable achievements to demonstrate impact.

                5.Word Choice: A score (0-100) on how well strong action words are used in the resume.
                Suggestions: Replacing weak verbs with stronger action words for better impact.

                6.Length: A score (0-100) evaluating whether the resume is the ideal length for the candidate’s experience level (typically one page for early-career, up to two pages for more experienced candidates).
                Suggestions: How to adjust the length for better conciseness or detail.

                7.Contact Info: A checklist of whether the resume includes complete contact information (phone and email).

                8.Missing Information: Identify any missing contact details.

                9.Completeness: A checklist of the necessary sections in the resume (e.g., Contact Info, Summary, Skills, Work Experience).

                10.Final thoughts.
                Suggestions: What sections may be missing or need further detail in easy english.
                
                11. Percentage Match 
                
                """
                
            # Generate response using Gemini AI
            response = get_gemini_response(input_text, pdf_content, input_prompt)
            st.subheader("Response:")
            st.write(response)

            # Get LinkedIn job recommendations
            linkedin_job_url = get_linkedin_job_recommendations(input_text)
            st.subheader("LinkedIn Job Recommendations:")
            st.markdown(f"[Click here to view job recommendations on LinkedIn]({linkedin_job_url})")

            # Generate interview preparation questions
            st.subheader("Interview Preparation Questions:")
            interview_questions = generate_interview_questions(input_text)
            st.write(interview_questions)

        elif linkedin_url:
            linkedin_data = get_linkedin_data(linkedin_url)
            st.subheader("LinkedIn Profile Data:")
            st.write(linkedin_data)
        else:
            st.warning("Please upload the resume or provide a LinkedIn URL to proceed.")

elif user_type == "Recruiter":
            with st.sidebar:
                # Recruiter Interface
                st.subheader("Recruiter Interface")
                job_description = st.text_area("Job Description: ", key="job_desc")
                uploaded_files = st.file_uploader("Upload Resumes (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if len(uploaded_files) > 20:
            st.warning("You can upload a maximum of 20 resumes.")

submit_recruiter = st.button("Analyze Resumes")

if submit_recruiter and uploaded_files:
            resume_data = []
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.name.split('.')[-1]
                chunks, faiss_index, resume_embeddings = process_document_with_faiss(uploaded_file, file_type)

                # Generate embeddings for the job description
                job_desc_embedding = embedder.encode([job_description], convert_to_tensor=True).cpu().numpy()

                # Calculate cosine similarity between job description and each resume chunk
                cosine_similarities = cosine_similarity(job_desc_embedding, resume_embeddings.cpu().numpy())

                # Calculate the average cosine similarity to determine the match percentage
                average_similarity = np.mean(cosine_similarities)

                if average_similarity >= 0.35:
                    fit_category = "Best Fit"
                elif 0.20 <= average_similarity < 0.34:
                    fit_category = "Moderate Fit"
                else:
                    fit_category = "Not Fit"

                # Collect resume data
                resume_data.append({
                    'name': uploaded_file.name,
                    'fit_category': fit_category,
                })

            # Sort resumes by fit category (Best Fit > Moderate Fit > Not Fit)
            ranked_resumes = sorted(resume_data, key=lambda x: ["Best Fit", "Moderate Fit","Not Fit" ].index(x['fit_category']))

            # Display ranked resumes in a table format
            st.subheader("Ranked Resumes:")
            st.table({
                "Resume Name": [resume['name'] for resume in ranked_resumes],
                "Fit Category": [resume['fit_category'] for resume in ranked_resumes]
            })

            

                

            

    
# Footer
st.markdown("""
<div class="footer">
    <p>&copy; Applicant Tracking System | Developed by <a href="https://www.linkedin.com/in/udaykiran-bawage-657754233">Udaykiran Bawage</p>
</div>
""", unsafe_allow_html=True)
