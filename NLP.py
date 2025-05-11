import spacy
import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re

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

# Define the learned data file path
learned_data_path = os.path.join(dataset_folder, 'learned_data.json')

# Load or create the learned data file
def load_learned_data():
    try:
        if os.path.exists(learned_data_path):
            with open(learned_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"facts": [], "job_info": {}}
    except Exception as e:
        logging.error(f"Error loading learned data: {str(e)}")
        return {"facts": [], "job_info": {}}

# Save learned data
def save_learned_data(data):
    try:
        with open(learned_data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logging.error(f"Error saving learned data: {str(e)}")
        return False

# Improved detection of teaching intent with more precise patterns
def is_teaching_intent(query):
    # Normalize query
    normalized_query = query.lower().strip()
    
    # Skip if the query contains question words at the beginning
    question_starters = ["what", "how", "why", "where", "when", "who", "which", "can", "do", "could", "would", "will", "should"]
    query_words = normalized_query.split()
    if query_words and query_words[0] in question_starters:
        return False, None
    
    # Skip if query contains question marks
    if "?" in normalized_query:
        return False, None
    
    # More precise teaching patterns that won't match regular questions
    teaching_patterns = [
        r'^remember that (.*)',
        r'^learn that (.*)',
        r'^know that (.*)',
        r'^note that (.*)',
        r'^(.*) is defined as (.*)',
        r'^(.*) are defined as (.*)',
        r'^a (.*) is (.*)',
        r'^an (.*) is (.*)',
        r'^the (.*) means (.*)',
        r'^(.*) refers to (.*)'
    ]
    
    for pattern in teaching_patterns:
        match = re.search(pattern, normalized_query)
        if match:
            return True, match.groups()
    
    # Pattern for "X is Y" but with additional checks to avoid matching questions
    is_pattern = r'^([^?]+) is ([^?]+)$'
    is_match = re.search(is_pattern, normalized_query)
    if is_match and not any(q in normalized_query for q in ["what is", "where is", "how is", "who is", "why is"]):
        return True, is_match.groups()
    
    # Pattern for "X are Y" but with similar checks
    are_pattern = r'^([^?]+) are ([^?]+)$'
    are_match = re.search(are_pattern, normalized_query)
    if are_match and not any(q in normalized_query for q in ["what are", "where are", "how are", "who are", "why are"]):
        return True, are_match.groups()
    
    return False, None

# Extract and store job-related learned information
def learn_job_information(query, groups):
    learned_data = load_learned_data()
    
    # Try to identify if this is job-related information
    job_keywords = ["job", "career", "profession", "role", "occupation"]
    skill_keywords = ["skill", "ability", "competency", "expertise"]
    degree_keywords = ["degree", "education", "qualification", "diploma"]
    cert_keywords = ["certificate", "certification", "credential"]
    
    # Basic parsing logic - can be enhanced
    if len(groups) == 2:  # For patterns like "X is Y"
        subject, predicate = groups
        subject = subject.strip()
        predicate = predicate.strip()
        
        # Check if the teaching is about a job
        if any(keyword in query.lower() for keyword in job_keywords):
            # Initialize job entry if it doesn't exist
            if subject not in learned_data["job_info"]:
                learned_data["job_info"][subject] = {
                    "Skills": [],
                    "Degrees": [],
                    "Certificates": [],
                    "Roadmap": []
                }
            
            # Determine what category the information belongs to
            if any(keyword in query.lower() for keyword in skill_keywords):
                if predicate not in learned_data["job_info"][subject]["Skills"]:
                    learned_data["job_info"][subject]["Skills"].append(predicate)
            elif any(keyword in query.lower() for keyword in degree_keywords):
                if predicate not in learned_data["job_info"][subject]["Degrees"]:
                    learned_data["job_info"][subject]["Degrees"].append(predicate)
            elif any(keyword in query.lower() for keyword in cert_keywords):
                if predicate not in learned_data["job_info"][subject]["Certificates"]:
                    learned_data["job_info"][subject]["Certificates"].append(predicate)
            else:
                # Default to a general fact if category can't be determined
                fact = f"{subject} {predicate}"
                if fact not in learned_data["facts"]:
                    learned_data["facts"].append(fact)
        else:
            # Store as a general fact
            fact = f"{subject} is {predicate}"
            if fact not in learned_data["facts"]:
                learned_data["facts"].append(fact)
    else:
        # For single group patterns like "remember that X"
        fact = groups[0].strip()
        if fact not in learned_data["facts"]:
            learned_data["facts"].append(fact)
    
    save_learned_data(learned_data)
    return f"I've learned that {fact if len(groups) == 1 else groups[0]} {'is ' + groups[1] if len(groups) > 1 else ''}. Thank you for teaching me!"

# Check learned data for answers
def check_learned_data(query):
    learned_data = load_learned_data()
    
    # Check if query is about a job we've learned about
    jobs_learned = list(learned_data["job_info"].keys())
    if jobs_learned:
        matched_job = get_best_matching_job(query, jobs_learned)
        if matched_job:
            request_type = identify_request_type(query)
            job_data = learned_data["job_info"][matched_job].copy()  # Use copy to avoid modifying original
            job_data["Job"] = matched_job
            
            # If we have information for this job, return it
            if any(len(job_data.get(k, [])) > 0 for k in ["Skills", "Degrees", "Certificates", "Roadmap"]):
                return format_response(job_data, request_type), []
    
    # Check general facts
    for fact in learned_data["facts"]:
        # Very basic relevance check - could be improved with NLP techniques
        if any(word in query.lower() for word in fact.lower().split() if len(word) > 3):
            return f"Based on what I've learned: {fact}", []
    
    return None, []

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


# Improved TF-IDF job matching for better results
def get_best_matching_job(query, job_list, threshold=0.15):  # Lower threshold for better matching
    if not job_list:
        return None
    
    # Clean and normalize query
    query = query.lower().strip()
    
    # Extract potential job titles from query
    job_related_phrases = [
        "i want to be a", "i want to be an", "i'm interested in", 
        "career as a", "career as an", "work as a", "work as an",
        "becoming a", "becoming an", "how to become a", "how to become an",
        "degrees for", "skills for", "certificates for", "roadmap for"
    ]
    
    cleaned_query = query
    for phrase in job_related_phrases:
        if phrase in query:
            # Extract text after the phrase
            cleaned_query = query.split(phrase, 1)[1].strip()
            break
    
    # Remove any category words from the query
    categories = ["degrees", "certificates", "roadmap", "skills", "career", "job"]
    query_words = cleaned_query.lower().split()
    filtered_query = " ".join([word for word in query_words if word not in categories])
    
    # Direct match attempt first (case insensitive)
    for job in job_list:
        if job.lower() == filtered_query.lower():
            return job
    
    # Check if query contains the job title exactly
    for job in job_list:
        if job.lower() in filtered_query.lower():
            return job
    
    # Check if job title contains the query
    for job in job_list:
        if filtered_query.lower() in job.lower():
            return job
    
    # Word-level matching (if query has multiple words that match parts of job titles)
    if len(filtered_query.split()) > 1:
        for job in job_list:
            # Count how many words in the query match words in the job title
            job_words = job.lower().split()
            query_words = filtered_query.lower().split()
            matches = sum(1 for word in query_words if any(word in job_word for job_word in job_words))
            
            # If at least half the words match
            if matches >= len(query_words) / 2:
                return job
    
    # If no direct matches, use TF-IDF as a fallback
    try:
        job_texts = [job.lower() for job in job_list]
        vectorizer = TfidfVectorizer().fit_transform(job_texts + [filtered_query.lower()])
        similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1])
        best_match_index = similarities.argmax()
        best_score = similarities[0][best_match_index]
        
        logging.info(f"Best match: {job_list[best_match_index]} with score {best_score}")
        
        if best_score >= threshold:
            return job_list[best_match_index]
    except Exception as e:
        logging.error(f"Error in TF-IDF matching: {str(e)}")
    
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


# Improved function to identify request type with better detection
def identify_request_type(query):
    query = query.lower()
    
    # Check for specific phrases requesting certain types of information
    degree_phrases = ["degree", "degrees", "education", "study", "college", "university", "academic", "qualification"]
    cert_phrases = ["certificate", "certificates", "cert", "certs", "certification", "credentials", "qualifications"]
    roadmap_phrases = ["roadmap", "path", "steps", "how to become", "career path", "journey", "progression"]
    skill_phrases = ["skill", "skills", "abilities", "competencies", "what do i need to know", "expertise"]
    
    # Count occurrences of phrases for each category
    degree_count = sum(1 for phrase in degree_phrases if phrase in query)
    cert_count = sum(1 for phrase in cert_phrases if phrase in query)
    roadmap_count = sum(1 for phrase in roadmap_phrases if phrase in query)
    skill_count = sum(1 for phrase in skill_phrases if phrase in query)
    
    # Return the category with the most matches
    counts = {
        "degrees": degree_count,
        "certificates": cert_count,
        "roadmap": roadmap_count,
        "skills": skill_count
    }
    
    # If there's a clear winner, return it
    max_category = max(counts, key=counts.get)
    if counts[max_category] > 0:
        return max_category
    
    # Default to None if no category is detected
    return None


# Enhanced process_query to better handle various input formats
def process_query(query):
    # First check if this is a teaching intent
    is_teaching, groups = is_teaching_intent(query)
    if is_teaching and groups:
        response = learn_job_information(query, groups)
        return response, ["Tell me more", "What else can you learn?"]
    
    # Continue with existing logic but still check learned data
    request_type = identify_request_type(query)
    logging.info(f"Request type identified: {request_type}")

    # Get all jobs from existing datasets
    all_jobs = []
    if not certificates_df.empty:
        all_jobs.extend(certificates_df['Job'].dropna().unique().tolist())
    if not job_degree_df.empty:
        all_jobs.extend(job_degree_df['Job'].dropna().unique().tolist())
    if not roadmaps_df.empty:
        all_jobs.extend(roadmaps_df['Job'].dropna().unique().tolist())
    for item in skills_data:
        if 'Job' in item:
            all_jobs.append(item['Job'])
    
    # Add learned jobs to the list of all jobs
    learned_data = load_learned_data()
    all_jobs.extend(list(learned_data["job_info"].keys()))
    
    # Remove duplicates
    all_jobs = list(set(all_jobs))
    
    # Try to find job title in the query
    matched_job = get_best_matching_job(query, all_jobs)

    if not matched_job:
        # Check if any general facts match
        learned_response, suggestions = check_learned_data(query)
        if learned_response:
            return learned_response, suggestions
        
        return "Sorry, I couldn't find a suitable career match for your input. Please try asking more clearly or with a career in mind.", [
            "Tell me about Software Engineer", 
            "What skills do I need for Data Scientist?",
            "Let me teach you something"
        ]

    logging.info(f"Matched job: {matched_job}")
    
    # Combine data from both sources
    job_data = get_job_related_data(matched_job)
    
    # Check if we have learned data for this job
    if matched_job in learned_data["job_info"]:
        learned_job_data = learned_data["job_info"][matched_job]
        
        # Merge learned data with existing data
        for key in ["Skills", "Degrees", "Certificates", "Roadmap"]:
            if key in learned_job_data and learned_job_data[key]:
                if key not in job_data:
                    job_data[key] = []
                job_data[key].extend(learned_job_data[key])
                # Remove duplicates
                job_data[key] = list(set(job_data[key]))

    # Generate related questions as suggestions
    suggestions = [
        f"What skills do I need for {matched_job}?",
        f"What degrees are recommended for {matched_job}?",
        f"What certificates are useful for {matched_job}?"
    ]
    
    return format_response(job_data, request_type), suggestions


# Optional CLI test
if __name__ == '__main__':
    user_query = input("Enter your career interest: ")
    response, suggestions = process_query(user_query)
    print(response)
    if suggestions:
        print("\nYou might also be interested in:")
        for suggestion in suggestions:
            print(f"- {suggestion}")
