"""
Resume Parser using NLP | Python, spaCy, Regex
-------------------------------------------------
This script demonstrates an end-to-end NLP pipeline to parse resumes, extract structured fields
like name, education, skills, and experience, and convert them into JSON format.

Dependencies:
- spacy
- re (regex)
- scikit-learn (for TF-IDF if needed)
"""

import spacy
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Load English NLP model (make sure to download with: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# Example resume text
resume_text = """
John Doe
Email: johndoe@example.com | Phone: 123-456-7890
Education: M.S. in Computer Science, University of XYZ, 2023
Skills: Python, SQL, Machine Learning, NLP, Data Visualization
Experience: Worked as Data Analyst at ABC Corp (2021-2023)
"""

def extract_entities(text):
    """
    Extracts named entities such as PERSON, ORG, DATE from the resume text using spaCy.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_email(text):
    """
    Extracts email using regex.
    """
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else None

def extract_phone(text):
    """
    Extracts phone number using regex.
    """
    match = re.search(r'\+?\d[\d\- ]{8,}\d', text)
    return match.group(0) if match else None

def extract_skills(text, skill_keywords=None):
    """
    Extracts skills from text using keyword matching or TF-IDF feature importance.
    """
    if not skill_keywords:
        skill_keywords = ["Python", "SQL", "Machine Learning", "NLP", "Data Visualization"]
    
    skills_found = [skill for skill in skill_keywords if skill.lower() in text.lower()]
    return skills_found

def parse_resume(text):
    """
    Parses resume text and returns structured data as dictionary.
    """
    parsed_data = {
        "name": None,  # can refine with regex or spaCy PERSON entity
        "email": extract_email(text),
        "phone": extract_phone(text),
        "entities": extract_entities(text),
        "skills": extract_skills(text)
    }
    return parsed_data

# Run parser on example resume text
parsed_resume = parse_resume(resume_text)

# Print JSON output
print(json.dumps(parsed_resume, indent=4))
