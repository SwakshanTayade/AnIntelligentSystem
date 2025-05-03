

#SET UP:

# 1. INSTALL BELOW LIBRARIES

        #pip install -r requirements.txt

        #not required 
        
        # pip install nltk

        # pip install spacy==2.3.5

        # pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

        # pip install pyresparser

# 2. CREAT A FOLDER AND NAME IT (e.g. resume)
        #2.1 create two more folders inside this folder (Logo and Uploaded_Resumes)
        #2.2 create two python files (App.py and Courses.py)

# 3. START YOUR SQL DATABASE
        #3.1 create a database named cv
        #then add the password of you sql database in the code below

# 4. CONTINUE WITH THE FOLLOWING CODE...

import streamlit as st
import pandas as pd
import base64,random
import time,datetime
#libraries to parse the resume pdf files
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io,random
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import resume_videos,interview_videos
import plotly.express as px #to create visualisations at the admin session
import nltk
import spacy
import yt_dlp
import os
from yt_dlp import YoutubeDL
from resume_builder import ResumeBuilder
import base64
from streamlit_lottie import st_lottie
import requests
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import os
import cv2
import numpy as np
from dotenv import load_dotenv
load_dotenv()
#creating instance of the resume builder;
resume_builder = ResumeBuilder()
nlp = spacy.load('en_core_web_sm')
print("NLpSwakshan ",nlp.pipe_names)

nltk.download('stopwords')
os.environ["PAFY_BACKEND"] = "yt-dlp"
import pafy #for uploading youtube videos

def load_lottie(url_or_path: str, sidebar: bool = False):
    """
    Load a Lottie animation from a URL or local file path and display it in Streamlit.

    Parameters:
        url_or_path (str): The URL or file path to the Lottie animation JSON file.
        sidebar (bool): If True, display the animation in the sidebar. Default is False.
    """
    if url_or_path.startswith(("http://", "https://")):
        # Load Lottie animation from a URL
        response = requests.get(url_or_path)
        if response.status_code != 200:
            st.error(f"Failed to load Lottie animation from URL: {url_or_path}")
            return None
        lottie_json = response.json()
    else:
        # Load Lottie animation from a local file
        try:
            import json
            with open(url_or_path, "r") as f:
                lottie_json = json.load(f)
        except Exception as e:
            st.error(f"Failed to load Lottie animation from file: {url_or_path}. Error: {e}")
            return None

    # Display the Lottie animation in the sidebar or main area
    if sidebar:
        with st.sidebar:
            st_lottie(lottie_json, height=200, key="sidebar_lottie")
    else:
        st_lottie(lottie_json, height=300, key="main_lottie")
def fetch_yt_video(link):
    ydl_opts = {}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=False)
        return info['title']

# url = "https://www.youtube.com/watch?v=VBK7BUSsrig"
# video = pafy.new(url)

# print("Title:", video.title)
# print("Duration:", video.duration)
# print("View Count:", video.viewcount)

import requests
import time

# Lightcast API credentials
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TOKEN_URL = "https://auth.emsicloud.com/connect/token"

# Cache the token and its expiry time
token_cache = {
    "access_token": None,
    "expiry_time": 0
}

def get_access_token():
    """
    Get a valid access token (either from cache or by generating a new one).
    """
    global token_cache
    
    # Check if the cached token is still valid
    if token_cache["access_token"] and time.time() < token_cache["expiry_time"]:
        return token_cache["access_token"]
    
    # Generate a new token
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": "emsi_open"
    }
    
    response = requests.post(TOKEN_URL, data=payload)
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data.get("access_token")
        token_cache["access_token"] = access_token
        token_cache["expiry_time"] = time.time() + 3600  # Token expires in 1 hour
        return access_token
    else:
        st.error("Failed to get access token. Status code: " + str(response.status_code))
        st.error("Response: " + response.text)
        return None

def get_table_download_link(df,filename,text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download Report</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def extract_text_from_scanned_pdf(pdf_path):
    """
    Extracts text from a scanned PDF using OCR with image preprocessing.
    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a string.
    """
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)

        # Initialize extracted text
        extracted_text = ""

        # Process each image
        for image in images:
            # Convert PIL image to OpenCV format
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Preprocessing the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)  # Apply OTSU thresholding
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))  # Define kernel for dilation
            dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)  # Apply dilation

            # Find contours
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Loop through contours and extract text
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cropped = img[y:y + h, x:x + w]  # Crop the text block
                text = pytesseract.image_to_string(cropped, lang='eng')  # Extract text using Tesseract
                extracted_text += text + "\n"
        
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from scanned PDF: {e}")
        return None

def pdf_reader(file):
    
    try:
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        
        with open(file, 'rb') as fh:
            for page in PDFPage.get_pages(fh,
                                        caching=True,
                                        check_extractable=True):
                page_interpreter.process_page(page)
                print(page)
            text = fake_file_handle.getvalue()

        if not text.strip():
            st.warning("No text found. Trying OCR...")
            with st.spinner("Extracting text using OCR..."):  # Show spinner while OCR is running
                ocr_text = extract_text_from_scanned_pdf(file)
                if ocr_text:
                    text = ocr_text
                else:
                    st.error("Failed to extract text using OCR.")
                    return None
        
        # close open handles
        converter.close()
        fake_file_handle.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 🎓**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5, key=f"slider_{random.randint(0, 100000)}")
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


def fetch_related_skills(skill, access_token):
    """
    Fetch related skills from the Lightcast Open Skills API.
    """
    SKILLS_API_URL = "https://emsiservices.com/skills/versions/latest/skills"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    params = {
        "q": skill,
        "limit": 10
    }
    
    try:
        response = requests.get(SKILLS_API_URL, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            related_skills = [skill['name'] for skill in data.get('data', [])]
            return related_skills
        else:
            st.error(f"Failed to fetch data from API. Status code: {response.status_code}")
            st.error("Response: " + response.text)
            return []
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []


import re
from datetime import datetime

def extract_name_from_email(email):
    """
    Extract name components from an email address
    Example: john.doe@example.com -> John Doe
    """
    if not email:
        return None
    
    # Remove domain part
    local_part = email.split('@')[0] if '@' in email else email
    
    # Common patterns for extracting name from email
    # Try to split by dots, underscores, or numbers
    name_parts = re.split(r'[._0-9]+', local_part)
    
    # Capitalize each part and join with space
    name = ' '.join([part.capitalize() for part in name_parts if part])
    
    return name if name else None

def extract_experience(resume_text):
    """
    Enhanced experience extraction with multiple pattern matching
    """
    experience_section = []
    total_experience_months = 0
    
    # Multiple patterns to detect experience section
    experience_patterns = [
        r'(?i)(?:experience|work history|professional experience|employment history|work experience)',
        r'(?i)(?:jobs?|positions?|roles?) held',
        r'(?i)(?:professional|work|employment) background'
    ]
    
    # Find experience section using multiple patterns
    exp_section_text = ""
    for pattern in experience_patterns:
        match = re.search(pattern, resume_text)
        if match:
            start = match.start()
            # Look for end of section (double newline or next major section)
            end = re.search(r'\n\s*\n|\b(?:education|skills|projects)\b', resume_text[start:], re.I)
            if end:
                exp_section_text = resume_text[start:start+end.start()]
            else:
                exp_section_text = resume_text[start:]
            break
    
    if not exp_section_text:
        return [], 0
    
    # Enhanced job entry patterns
    job_patterns = [
        # Pattern 1: Job title at Company (MM/YYYY - MM/YYYY)
        r'(?i)([\w\s\-\.]+)\s*(?:at|@|in)\s*([\w\s\-\.,&]+)\s*\((\d{1,2}/\d{4}|\w+\s\d{4})\s*-\s*(\d{1,2}/\d{4}|\w+\s\d{4}|present|current)\)',
        # Pattern 2: Company - Job title (MM/YYYY - MM/YYYY)
        r'(?i)([\w\s\-\.,&]+)\s*[-–]\s*([\w\s\-\.]+)\s*\((\d{1,2}/\d{4}|\w+\s\d{4})\s*-\s*(\d{1,2}/\d{4}|\w+\s\d{4}|present|current)\)',
        # Pattern 3: Bullet point style
        r'(?i)^\s*[-•*]\s*([\w\s\-\.]+)\s*(?:at|@|in)\s*([\w\s\-\.,&]+)\s*,\s*(\d{1,2}/\d{4}|\w+\s\d{4})\s*-\s*(\d{1,2}/\d{4}|\w+\s\d{4}|present|current)',
    ]
    
    for pattern in job_patterns:
        for match in re.finditer(pattern, exp_section_text):
            job_title, company, start_date, end_date = match.groups()
            
            # Standardize date formats
            try:
                if len(start_date.split('/')) == 2:
                    start_date = datetime.strptime(start_date, '%m/%Y')
                else:
                    start_date = datetime.strptime(start_date, '%B %Y')
                
                if end_date.lower() in ['present', 'current']:
                    end_date = datetime.now()
                elif len(end_date.split('/')) == 2:
                    end_date = datetime.strptime(end_date, '%m/%Y')
                else:
                    end_date = datetime.strptime(end_date, '%B %Y')
                
                duration = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                total_experience_months += duration
                
                experience_section.append({
                    'job_title': job_title.strip(),
                    'company': company.strip(),
                    'start_date': start_date.strftime('%b %Y'),
                    'end_date': end_date.strftime('%b %Y'),
                    'duration_months': duration,
                    'duration_years': round(duration / 12, 1)
                })
            except ValueError as e:
                continue
    
    total_experience_years = round(total_experience_months / 12, 1)
    return experience_section, total_experience_years


def determine_experience_level(total_experience_years):
    """
    Determine candidate level based on actual experience years
    """
    if total_experience_years == 0:
        return "Fresher"
    elif total_experience_years <= 3:
        return "Intermediate"
    elif total_experience_years <= 7:
        return "Experienced"
    else:
        return "Senior"

def fetch_youtube_courses(query, max_results=1):
    """
    Fetch YouTube videos (courses) based on a search query.
    :param query: The search query (e.g., "data science course").
    :param max_results: Maximum number of results to fetch.
    :return: List of course titles and URLs.
    """
    ydl_opts = {
        'extract_flat': True,  # Extract metadata without downloading
        'quiet': True,         # Suppress output
        'default_search': 'ytsearch',  # Search on YouTube
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            search_results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            courses = []
            for entry in search_results['entries']:
                title = entry.get('title', 'No Title')
                url = entry.get('url', '')
                courses.append([title, url])
            return courses
        except Exception as e:
            st.error(f"Failed to fetch YouTube courses: {e}")
            return []

def analyze_resume_with_hf(resume_text, hf_token):
    """
    Analyze resume using Hugging Face's Inference API.
    Returns a score between 0-100 using zero-shot classification approach.
    """
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer {hf_token}"}

    # Define scoring categories with score ranges
    score_categories = [
        "Excellent (90-100): Complete sections, strong keywords, professional tone, clear achievements, excellent structure",
        "Good (75-89): Mostly complete, relevant keywords, professional tone, some achievements, good structure",
        "Average (50-74): Some missing sections, some relevant keywords, decent tone, few achievements, basic structure",
        "Weak (0-49): Incomplete, weak keywords, unprofessional tone, no achievements, poor structure"
    ]

    payload = {
        "inputs": resume_text[:2000],  # Limit input size
        "parameters": {
            "candidate_labels": score_categories,
            "multi_label": False
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # Raises exception for 4XX/5XX errors
        
        result = response.json()
        
        # Extract the predicted category
        predicted_category = result['labels'][0]
        confidence = result['scores'][0]
        
        # Map category to numerical score (midpoint of range)
        if "Excellent" in predicted_category:
            return min(100, int(95 * confidence))  # Scale by confidence
        elif "Good" in predicted_category:
            return min(89, int(82 * confidence))
        elif "Average" in predicted_category:
            return min(74, int(62 * confidence))
        else:
            return min(49, int(30 * confidence))
            
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return 69  # Fallback average score
    except (KeyError, IndexError, ValueError) as e:
        st.error(f"Error parsing API response: {str(e)}")
        return 67
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return 68



def detect_sections(resume_text):
    doc = nlp(resume_text)
    sections = {
        'Objective': ['objective', 'summary', 'career goal'],
        'Skills': ['skills', 'technical skills', 'proficiencies'],
        'Experience': ['experience', 'work history', 'employment'],
        'Education': ['education', 'academic'],
        'Projects': ['projects', 'portfolio'],
        'Achievements': ['achievements', 'awards', 'honors'],
        'Certifications': ['certifications', 'certificates', 'credentials']
    }
    section_scores = {}
    for section, keywords in sections.items():
        section_scores[section] = any(keyword.lower() in resume_text.lower() for keyword in keywords)
    return section_scores
#CONNECT TO DATABASE

connection = pymysql.connect(host='localhost',user='root',password='Swakshan@123',db='cv')
cursor = connection.cursor()

def insert_data(name,email,res_score,timestamp,no_of_pages,reco_field,cand_level,skills,recommended_skills,courses):
    DB_table_name = 'user_data'
    insert_sql = "insert into " + DB_table_name + """
    values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (name, email, str(res_score), timestamp,str(no_of_pages), reco_field, cand_level, skills,recommended_skills,courses)
    cursor.execute(insert_sql, rec_values)
    connection.commit()

st.set_page_config(
   page_title="ResumeForge.ai",
   page_icon='./Logo/Logo/logo2.png',
)
def run():
    #load_lottie to the st.sidebar
    
    
    load_lottie("https://assets10.lottiefiles.com/packages/lf20_3rwasyjy.json")
    load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json", sidebar=True)
    st.title("ResumeForge.ai")

    activities = ["Resume Analyzer", "Developer","Resume Builder"]
    choice = st.sidebar.segmented_control('Choose the Mode', options=activities)
    link = '[©Developed With ♥️](https://www.linkedin.com/in/swakshan-tayade-3a0a5b235/)'
    st.sidebar.markdown(link, unsafe_allow_html=True)

    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)

    # Create table
    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """(
                    ID INT NOT NULL AUTO_INCREMENT,
                    Name varchar(500) NOT NULL,
                    Email_ID VARCHAR(500) NOT NULL,
                    resume_score VARCHAR(8) NOT NULL,
                    Timestamp VARCHAR(50) NOT NULL,
                    Page_no VARCHAR(5) NOT NULL,
                    Predicted_Field TEXT NOT NULL,  -- Changed from BLOB to TEXT
                    User_level TEXT NOT NULL,       -- Changed from BLOB to TEXT
                    Actual_skills TEXT NOT NULL,    -- Changed from BLOB to TEXT
                    Recommended_skills TEXT NOT NULL, -- Changed from BLOB to TEXT
                    Recommended_courses TEXT NOT NULL, -- Changed from BLOB to TEXT
                    PRIMARY KEY (ID)
                     );
                    """
    cursor.execute(table_sql)
    if choice == 'Resume Analyzer':
        st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload your resume, and get smart recommendations</h5>''',
                    unsafe_allow_html=True)
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        
        #section of resume builder;
            
            
        
        if pdf_file is not None:
            with st.spinner('Uploading your Resume...'):
                time.sleep(4)
            save_image_path = './Uploaded_Resumes/'+pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            if resume_data:
                ## Get the whole resume data
                resume_text = pdf_reader(save_image_path)
                experience_section, total_experience_years = extract_experience(resume_text)
                resume_data['experience'] = experience_section
                resume_data['total_experience_years'] = total_experience_years
                print("Extracted Resume Text (First 500 characters):", resume_text[:500])  # Prints first 500 characters
                st.text_area("Extracted Resume Text:", resume_text, height=300)  # Displays full text in Streamlit UI   
                st.header("**Resume Analysis**")
                # Check if 'name' exists and is not None
                # Extract name with improved function
                resume_data['name'] = extract_name_from_email(resume_data.get('email'))
                if not resume_data.get('name'):
                    st.success("Hello!")
                else:
                    st.success("Hello " + resume_data['name'])

                st.subheader("**Your Basic info**")
                try:
                    st.text('Name: ' + (resume_data['name'] if resume_data.get('name') else 'Not provided'))
                    st.text('Email: ' + (resume_data['email'] if resume_data.get('email') else 'Not provided'))
                    st.text('Contact: ' + (resume_data['mobile_number'] if resume_data.get('mobile_number') else 'Not provided'))
                    st.text('Resume pages: ' + str(resume_data['no_of_pages'] if resume_data.get('no_of_pages') else 'Not provided'))
                except Exception as e:
                    st.error(f"Error displaying basic info: {e}")
                    
                # Determine experience level properly
                cand_level = determine_experience_level(total_experience_years)

                # Display to user
                st.text(f'Total Experience: {total_experience_years} years')
                if experience_section:
                    st.subheader("Work Experience Details")
                    for exp in experience_section:
                        st.text(f"{exp['job_title']} at {exp['company']} ({exp['start_date']} - {exp['end_date']})")
                # st.subheader("**Skills Recommendation💡**")
                ## Skill shows
                keywords = st_tags(label='### Your Current Skills',
                text='See our skills recommendation below',
                    value=resume_data['skills'],key = '1  ')

                ##  keywords
                keywords = {
                    'ds_dev': ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'flask', 'streamlit'],
                    'ui_ux_designer': ["UI/UX", "Wireframing", "Prototyping", "User Research", "Adobe XD",
                                    "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'backend_dev': ["Python", "Java", "Node.js", "SQL", "APIs", "Django", "Flask", "Database Design",
                                    "Microservices", "Docker", "Analytical thinking", "Problem-solving", "Team collaboration", "Communication"],
                    'mobile_app_dev': ["Swift", "Kotlin", "React Native", "Flutter", "Mobile UI/UX", "App Store Deployment",
                                    "User-centric thinking", "Problem-solving", "Attention to detail"],
                    'ios_dev': ["Swift", "Objective-C", "Xcode", "iOS SDK", "App Store Deployment", "User-centric thinking",
                                "Problem-solving", "Attention to detail"],
                    'game_dev': ["Unity", "Unreal Engine", "C++", "C#", "3D Graphics", "Game Physics", "Graphics Programming",
                                "Physics Simulation", "Multiplayer", "Creativity", "Problem-solving", "Team collaboration"],
                    'machine_learning': ["Python", "R", "Machine Learning", "Statistics", "SQL", "Deep Learning",
                                        "Analytical thinking", "Research", "Problem-solving"],
                    'data_analyst': ["SQL", "Excel", "Python", "Data Visualization", "Statistics",
                                    "Analytical thinking", "Research", "Problem-solving"],
                    'cloud_engineer': ["AWS", "Azure", "Google Cloud", "DevOps", "Kubernetes", "Docker",
                                    "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'devops_engineer': ["CI/CD", "Docker", "Kubernetes", "Jenkins", "Ansible", "Git",
                                        "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'site_reliability_engineer': ["Linux", "Networking", "Monitoring", "Incident Response", "Automation", "Scripting",
                                                "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'cybersecurity_engineer': ["Firewalls", "Intrusion Detection", "Incident Response", "Encryption", "Penetration Testing",
                                            "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'penetration_tester': ["Penetration Testing", "Ethical Hacking", "Vulnerability Assessment", "Security Tools", "Networking",
                                        "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'network_engineer': ["Networking", "Cisco", "Routing & Switching", "Firewalls", "Troubleshooting",
                                        "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'software_engineer': ["Algorithms", "Data Structures", "Object-Oriented Programming", "Design Patterns", "Testing",
                                        "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'database_admin': ["SQL", "Database Design", "Performance Tuning", "Backup & Recovery", "Security",
                                    "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'business_analyst': ["SQL", "Excel", "Data Analysis", "Requirements Gathering", "Data Visualization",
                                        "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'product_manager': ["Product Development", "Market Research", "Roadmapping", "Stakeholder Management", "Agile",
                                        "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'technical_writer': ["Technical Writing", "Documentation", "Editing", "Proofreading", "Research",
                                        "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'it_support': ["Troubleshooting", "Help Desk", "Networking", "Hardware", "Software",
                                "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'data_engineer': ["ETL", "Data Warehousing", "Big Data", "SQL", "Python",
                                    "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'ai_engineer': ["Machine Learning", "Deep Learning", "Neural Networks", "Python", "TensorFlow",
                                    "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'blockchain_engineer': ["Blockchain", "Smart Contracts", "Cryptocurrency", "DApps", "Solidity",
                                            "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'qa_engineer': ["Testing", "Automation", "Selenium", "JIRA", "Bug Tracking",
                                    "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'system_admin': ["Linux", "Windows", "Networking", "Security", "Troubleshooting",
                                    "Analytical thinking", "Problem-solving", "Team collaboration"]
                    
                }
                
                # Initialize variables
                recommended_skills = []
                reco_field = ''
                rec_course = ''
                category = None
                all_recommended_skills = []
                processed_categories = set()

                # Dictionary to count the number of matching skills for each category
                category_match_count = {cat: 0 for cat in keywords.keys()}

                # Count the number of matching skills for each category
                for i in resume_data['skills']:
                    print(f"Checking skill: {i.lower()}")  # Debug: Print the skill being checked
                    for cat, skills in keywords.items():
                        if i.lower() in [s.lower() for s in skills]:  # Ensure case-insensitive matching
                            print(f"Matched skill: {i.lower()} with category: {cat}")  # Debug: Print matched skill and category
                            category_match_count[cat] += 1  # Increment the match count for this category

                # Find the category with the maximum number of matching skills
                max_matches = max(category_match_count.values())
                if max_matches > 0:
                    # Get the category with the maximum matches
                    category = max(category_match_count, key=category_match_count.get)
                    st.success(f"** Our analysis says you are looking for {category.replace('_', ' ').title()} Jobs **")

                    # Collect recommended skills for the selected category
                    for i in resume_data['skills']:
                        if i.lower() in [s.lower() for s in keywords[category]]:  # Ensure case-insensitive matching
                            access_token = get_access_token()
                            if access_token:
                                related_skills = fetch_related_skills(i.lower(), access_token)
                                print(f"Related skills from API: {related_skills}")
                            else:
                                related_skills = []
                                st.error("Failed to fetch related skills from the API.")
                            
                            recommended_skills = keywords[category] + related_skills
                            all_recommended_skills.extend(recommended_skills)

                # Display all recommended skills at once
                if all_recommended_skills:
                    unique_key = f"all_recommended_skills"
                    recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                text='Recommended skills generated from System',
                                                value=all_recommended_skills, key=unique_key)
                    
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost🚀 your chances of getting a Job</h4>''',
                                unsafe_allow_html=True)
                else:
                    st.warning("No recommended skills found for this job role.")
                # Fetch YouTube courses for all recommended skills
                # max_videos = st.slider('Select the number of YouTube video recommendations:', 1, 20, 5)
                with st.spinner('Fetching YouTube course recommendations...'):  # Show spinner while fetching
                    youtube_courses = []
                    max_videos = 20  # Set the maximum number of videos to fetch
                    video_count = 0  # Counter to track the number of videos fetched

                    for skill in all_recommended_skills:
                        if video_count >= max_videos:  # Stop if the maximum number of videos is reached
                            break

                        query = f"{skill} course"
                        courses = fetch_youtube_courses(query, max_results=1)  # Fetch only 1 video per skill
                        if courses:
                            youtube_courses.extend(courses)
                            video_count += 1  # Increment the counter for each video fetched

                # Display YouTube courses
                if youtube_courses:
                    st.subheader(f"**YouTube Course Recommendations 🎓**")
                    for title, url in youtube_courses:
                        st.markdown(f"- [{title}]({url})")
                else:
                    st.warning("No YouTube courses found for the recommended skills.")

                # Recommend courses from the hardcoded list
                # if category is None:
                #     st.warning("No matching job category found for your skills. Using default recommendations.")
                #     category = 'ds_dev'
                # course = category + '_course'
                # rec_course = course_recommender(eval(course))

                ## Insert into table
                ts = time.time()
                cur_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)

                ### Resume writing recommendation
                st.subheader("**Resume Tips & Ideas💡**")
                section_scores = detect_sections(resume_text)
                for section, present in section_scores.items():
                    if present:
                        st.markdown(f'''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added {section}</h5>''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''<h5 style='text-align: left; color: red;'>[-] Please add {section} to improve your resume.</h5>''', unsafe_allow_html=True)

                st.subheader("**AI Resume Score Analysis**")
                try:
                    with st.spinner('Analyzing resume quality...'):
                        HF_TOKEN = os.getenv("HF_TOKEN")  # Replace with your actual HF token
                        resume_score = analyze_resume_with_hf(resume_text, HF_TOKEN)
                    
                    # Progress bar
                    my_bar = st.progress(0)
                    for percent_complete in range(resume_score):
                        time.sleep(0.02)
                        my_bar.progress(percent_complete*2 + 1)
                    
                    st.success(f'**AI Assessment Score: {resume_score*2}/100**')
                    st.warning("""
                    **Scoring Criteria:**
                    - Completeness of sections  
                    - Keyword relevance  
                    - Professional tone  
                    - Demonstrated achievements  
                    - Structure  
                    """)
                except Exception as e:
                    st.error(f"Error analyzing resume: {e}")
                st.balloons()

                insert_data(resume_data['name'], resume_data['email'], str(resume_score), timestamp,
                            str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']),
                            str(recommended_skills), str(rec_course))

                ## Resume writing video
                st.header("**Bonus Video for Resume Writing Tips💡**")
                resume_vid = random.choice(resume_videos)
                res_vid_title = fetch_yt_video(resume_vid)
                st.subheader("✅ **"+res_vid_title+"**")
                st.video(resume_vid)

                ## Interview Preparation Video
                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = random.choice(interview_videos)
                int_vid_title = fetch_yt_video(interview_vid)
                st.subheader("✅ **" + int_vid_title + "**")
                st.video(interview_vid)

                connection.commit()
            else:
                st.error('Something went wrong..')
    elif choice == 'Resume Builder':
        if st.sidebar.button("📝 Build Your Resume"):
            st.session_state['resume_builder'] = True
            
        if 'resume_builder' in st.session_state and st.session_state['resume_builder']:
            st.header("Lets build a great resume")
        
            
            template = st.selectbox("Choose a Template", ["Modern", "Professional", "Minimal Creative"])
            # Collect resume data from the user
            personal_info = {}
            personal_info['full_name'] = st.text_input("Full Name")
            personal_info['email'] = st.text_input("Email")
            personal_info['phone'] = st.text_input("Phone")
            personal_info['location'] = st.text_input("Location")
            personal_info['linkedin'] = st.text_input("LinkedIn Profile")
            personal_info['portfolio'] = st.text_input("Portfolio Website")
            personal_info['title'] = st.text_input("Professional Title")
            
            summary  = st.text_area("Professional Summary")
            
            experience = []
            st.subheader("Experience")
            num_experience = st.number_input("Number of Experience Entries",min_value=0,max_value=10,value=1)
            
            for i in range(num_experience):
                exp = {}
                exp['position'] = st.text_input(f"Position {i+1}")
                exp['company'] = st.text_input(f"Company {i+1}")
                exp['start_date'] = st.text_input(f"Start Date {i+1}")
                exp['end_date'] = st.text_input(f"End Date {i+1}")
                exp['description'] = st.text_area(f"Description {i+1}")
                experience.append(exp)
                
            education = []
            st.header("Education")
            num_education = st.number_input("Education", min_value=1, max_value=10, value=1)
            
            for i in range (num_education):
                edu = {}
                edu['school'] = st.text_input(f"School {i+1}")
                edu['degree'] = st.text_input(f"Degree {i+1}")
                edu['field'] = st.text_input(f"Field of Study {i+1}")
                edu['graduation_date'] = st.text_input(f"Graduation Date {i+1}")
                edu['gpa'] = st.text_input(f"GPA {i+1}")
                education.append(edu)
            
            skills = {}
            st.subheader("Skills")
            skills['technical'] = st.text_area("Technical Skills (comma separated)")
            skills['soft'] = st.text_area("Soft Skills (comma separated)")
            skills['languages'] = st.text_area("Languages (comma separated)")
            skills['tools'] = st.text_area("Tools & Technologies (comma separated)")
            
            
            resume_data = None
            if st.button("Generate Resume"):
                resume_data = {
                    'personal_info': personal_info,
                    'summary': summary,
                    'experience': experience,
                    'education': education,
                    'skills': skills,
                    'template': template
                }
                
            if resume_data is not None:
                #generating resume;
                resume_buffer = resume_builder.generate_resume(resume_data)
                
                #downloadable resume
                st.success("Resume generated successfully")
                st.download_button(
                    label="Download Resume",
                    data=resume_buffer,
                    file_name="resume.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
    else:
        st.success('Welcome to Admin Side (FYP_G3)')
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            if ad_user == 'FYP_G3' and ad_password == '123':
                st.success("Welcome FYP_G3 !")
                # Fetch and decode data
                cursor.execute('''SELECT * FROM user_data''')
                data = cursor.fetchall()
                decoded_data = []
                for row in data:
                    decoded_row = list(row)
                    for i in range(6, 11):  # Decode BLOB columns
                        if isinstance(decoded_row[i], bytes):
                            decoded_row[i] = decoded_row[i].decode('utf-8')
                    decoded_data.append(decoded_row)
                
                # Create DataFrame
                df = pd.DataFrame(decoded_data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                                                        'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                                        'Recommended Course'])
                
                # Display DataFrame
                st.header("**User's Data**")
                st.dataframe(df)
                st.markdown(get_table_download_link(df, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)

                ## Pie chart for predicted field recommendations
                st.subheader("**Pie-Chart for Predicted Field Recommendation**")
                labels = df['Predicted Field'].unique()
                values = df['Predicted Field'].value_counts()
                fig = px.pie(df, values=values, names=labels, title='Predicted Field according to the Skills')
                st.plotly_chart(fig)

                ### Pie chart for User's Experienced Level
                st.subheader("**Pie-Chart for User's Experienced Level**")
                labels = df['User Level'].unique()
                values = df['User Level'].value_counts()
                fig = px.pie(df, values=values, names=labels, title="Pie-Chart📈 for User's👨‍💻 Experienced Level")
                st.plotly_chart(fig)
            else:
                st.error("Wrong ID & Password Provided")
run()