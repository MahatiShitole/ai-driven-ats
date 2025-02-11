# Applicant Tracking System (ATS)

An AI-powered **Applicant Tracking System (ATS)** designed to streamline the recruitment process by analyzing resumes, matching them with job descriptions, and providing actionable insights for both job seekers and recruiters.

## Features

- **Resume Analysis**: Upload resumes (PDF, DOCX, TXT) and analyze them for ATS compatibility, keyword matching, and formatting.
- **Job Description Matching**: Compare resumes with job descriptions to calculate match percentages and provide improvement suggestions.
- **LinkedIn Integration**: Fetch LinkedIn profile data and generate job recommendations based on job descriptions.
- **Interview Preparation**: Generate technical interview questions tailored to the job role (fresher, intermediate, expert levels).
- **Recruiter Dashboard**: Recruiters can upload multiple resumes, rank them based on job description fit, and categorize them as "Best Fit," "Moderate Fit," or "Not Fit."
- **AI-Powered Insights**: Leverages **Google Gemini AI** and **Sentence Transformers** for natural language processing and semantic analysis.
- **Personalized Skill Development** – Conducts skill gap analysis & suggests online courses to boost career growth.
## Technologies Used

- **Python Libraries**:
  - `streamlit`: For building the web app interface.
  - `sentence-transformers`: For generating embeddings and semantic similarity.
  - `faiss`: For efficient similarity search and indexing.
  - `PyPDF2`, `docx`, `fpdf`: For processing PDF, DOCX, and TXT files.
  - `google-generativeai`: For AI-powered text generation and analysis.
  - `plotly`, `pandas`: For data visualization and manipulation.
- **APIs**:
  - Google Gemini AI API for advanced text analysis.
  - LinkedIn (simulated) for profile data and job recommendations.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ai-driven-ats.git
   cd Applicant-Tracking-System
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```


3. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory.
   - Add your Google Gemini API key:
     ```plaintext
     GEMINI_API_KEY=your_api_key_here
     ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### For Job Seekers (Employees)
1. Select **Employee** as the user type.
2. Upload your resume (PDF, DOCX, or TXT).
3. Enter the job description and optionally provide your LinkedIn profile URL.
4. Click **See Results** to get a detailed analysis of your resume, including:
   - ATS compatibility score.
   - Keyword matching and missing keywords.
   - Suggestions for improving your resume.
5. Click **Make A Match** to generate interview questions and LinkedIn job recommendations.

### For Recruiters
1. Select **Recruiter** as the user type.
2. Enter the job description.
3. Upload multiple resumes (up to 20).
4. Click **Analyze Resumes** to rank the resumes based on their fit with the job description.
5. View the ranked resumes categorized as "Best Fit," "Moderate Fit," or "Not Fit."


## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to your branch.
4. Submit a pull request with a detailed description of your changes.




