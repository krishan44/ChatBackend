import spacy
import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logging.error(f"Error loading spaCy model: {str(e)}")
    raise

# Define the dataset folder path
dataset_folder = os.path.join(os.getcwd(), 'Dataset')


# Load Excel datasets
def load_excel_dataset(file_name):
    try:
        path = os.path.join(dataset_folder, file_name)
        return pd.read_excel(path)
    except Exception as e:
        logging.error(f"Error loading {file_name}: {str(e)}")
        return pd.DataFrame()


certificates_df = load_excel_dataset('Certificates.xlsx')
job_degree_df = load_excel_dataset('Degrees.xlsx')
roadmaps_df = load_excel_dataset('Roadmaps.xlsx')

# Load JSON dataset for skills
try:
    with open(os.path.join(dataset_folder, 'Skills.json'), 'r', encoding='utf-8') as f:
        skills_data = json.load(f)
except Exception as e:
    logging.error(f"Error loading Skills.json: {str(e)}")
    skills_data = []


# TF-IDF job matching
def get_best_matching_job(query, job_list, threshold=0.3):
    # Remove any category words from the query
    categories = ["degrees", "certificates", "roadmap", "skills"]
    query_words = query.lower().split()
    filtered_query = " ".join([word for word in query_words if word not in categories])

    job_texts = [job.lower() for job in job_list]
    vectorizer = TfidfVectorizer().fit_transform(job_texts + [filtered_query.lower()])
    similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score >= threshold:
        return job_list[best_match_index]
    else:
        return None


# Get job data
def get_job_related_data(job_title):
    result = {"Job": job_title}

    # Certificates
    cert_row = certificates_df[certificates_df['Job'].str.lower() == job_title.lower()]
    if not cert_row.empty:
        result['Certificates'] = cert_row.iloc[0, 1:].dropna().tolist()

    # Degrees (support multiple job aliases per row)
    for _, row in job_degree_df.iterrows():
        job_aliases = [j.strip().lower() for j in str(row['Job']).split(',')]
        if job_title.lower() in job_aliases:
            result['Degrees'] = row.iloc[1:].dropna().tolist()
            break

    # Roadmap
    roadmap_row = roadmaps_df[roadmaps_df['Job'].str.lower() == job_title.lower()]
    if not roadmap_row.empty:
        result['Roadmap'] = roadmap_row.iloc[0, 1:].dropna().tolist()

    # Skills
    for item in skills_data:
        if item['Job'].lower() == job_title.lower():
            result['Skills'] = [s['name'] for s in item['skills']]
            break

    return result


# Format response based on requested information
def format_response(job_data, requested_info=None):
    job = job_data.get("Job", "Unknown")

    # Check what information was requested
    if requested_info:
        # Handle specific information request
        if requested_info == "degrees":
            degrees = job_data.get("Degrees", [])
            if degrees:
                response = f"**📚 Recommended Degrees for {job}:**\n"
                for degree in degrees:
                    response += f"- {degree}\n"
                return response.strip()
            else:
                return f"No degree information available for {job}."

        elif requested_info == "certificates":
            certificates = job_data.get("Certificates", [])
            if certificates:
                response = f"**🎓 Useful Certificates for {job}:**\n"
                for cert in certificates:
                    response += f"- {cert}\n"
                return response.strip()
            else:
                return f"No certificate information available for {job}."

        elif requested_info == "roadmap":
            roadmap = job_data.get("Roadmap", [])
            if roadmap:
                response = f"**🗺️ Career Roadmap for {job}:**\n"
                for step in roadmap:
                    response += f"- {step}\n"
                return response.strip()
            else:
                return f"No roadmap information available for {job}."

        elif requested_info == "skills":
            skills = job_data.get("Skills", [])
            if skills:
                response = f"**🛠️ Key Skills for {job}:**\n"
                for skill in skills:
                    response += f"- {skill}\n"
                return response.strip()
            else:
                return f"No skills information available for {job}."

    # If no specific information was requested or the request wasn't recognized,
    # return the full guide
    response = f"Here's a guide to becoming a **{job}**:\n\n"

    if "Degrees" in job_data and job_data["Degrees"]:
        response += "**📚 Recommended Degrees:**\n"
        for degree in job_data["Degrees"]:
            response += f"- {degree}\n"
        response += "\n"

    if "Certificates" in job_data and job_data["Certificates"]:
        response += "**🎓 Useful Certificates:**\n"
        for cert in job_data["Certificates"]:
            response += f"- {cert}\n"
        response += "\n"

    if "Roadmap" in job_data and job_data["Roadmap"]:
        response += "**🗺️ Career Roadmap:**\n"
        for step in job_data["Roadmap"]:
            response += f"- {step}\n"
        response += "\n"

    if "Skills" in job_data and job_data["Skills"]:
        response += "**🛠️ Key Skills to Develop:**\n"
        for skill in job_data["Skills"]:
            response += f"- {skill}\n"

    return response.strip()


# Identify requested information type
def identify_request_type(query):
    query = query.lower()

    if "degree" in query or "degrees" in query:
        return "degrees"
    elif "certificate" in query or "certificates" in query or "cert" in query:
        return "certificates"
    elif "roadmap" in query or "path" in query or "steps" in query:
        return "roadmap"
    elif "skill" in query or "skills" in query:
        return "skills"

    return None


# NLP main function
def process_query(query):
    # Identify what information is being requested
    request_type = identify_request_type(query)
    logging.info(f"Request type identified: {request_type}")

    all_jobs = certificates_df['Job'].dropna().unique().tolist()
    matched_job = get_best_matching_job(query, all_jobs)

    if not matched_job:
        return "Sorry, I couldn't find a suitable career match for your input. Please try asking more clearly or with a career in mind."

    logging.info(f"Matched job: {matched_job}")
    job_data = get_job_related_data(matched_job)

    return format_response(job_data, request_type)


# Optional CLI test
if __name__ == '__main__':
    user_query = input("Enter your career interest: ")
    response = process_query(user_query)
    print(response)