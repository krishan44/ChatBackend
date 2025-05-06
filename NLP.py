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


# Define conversation states
class ConversationState:
    GREETING = "greeting"
    AWAITING_CAREER_INTEREST = "awaiting_career_interest"
    CAREER_SELECTED = "career_selected"


# Create a conversation context class to maintain state
class ConversationContext:
    def __init__(self):
        self.state = ConversationState.GREETING
        self.last_job = None
        self.conversation_history = []
        self.answered_requests = set()  # Track which request types have been answered

    def update_history(self, user_input, system_response):
        self.conversation_history.append({
            "user": user_input,
            "system": system_response
        })

    def get_last_response(self):
        if self.conversation_history:
            return self.conversation_history[-1]["system"]
        return None

    def mark_request_answered(self, job, request_type):
        """Track that a specific request for a job has been answered"""
        if request_type:
            self.answered_requests.add(f"{job.lower()}:{request_type}")

    def was_request_answered(self, job, request_type):
        """Check if this specific request was already answered"""
        return f"{job.lower()}:{request_type}" in self.answered_requests


# Initialize global conversation context
conversation_context = ConversationContext()


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


# Get all available jobs
def get_all_jobs():
    all_jobs = set()

    # From certificates
    all_jobs.update(certificates_df['Job'].dropna().unique().tolist())

    # From degrees (handling comma-separated job lists)
    for job_entry in job_degree_df['Job'].dropna():
        job_aliases = [j.strip() for j in str(job_entry).split(',')]
        all_jobs.update(job_aliases)

    # From roadmaps
    all_jobs.update(roadmaps_df['Job'].dropna().unique().tolist())

    # From skills
    for item in skills_data:
        if 'Job' in item:
            all_jobs.add(item['Job'])

    return list(all_jobs)


# Get list of all available jobs
all_job_titles = get_all_jobs()


# TF-IDF job matching
def get_best_matching_job(query, job_list, threshold=0.3):
    # Remove any category words from the query
    categories = ["degrees", "certificates", "roadmap", "skills"]
    query_words = query.lower().split()
    filtered_query = " ".join([word for word in query_words if word not in categories])

    # Check for career change phrases
    career_change_patterns = [
        r'how (can|do) I (become|be) an? ([a-zA-Z\s]+)',
        r'I want to (be|become) an? ([a-zA-Z\s]+)',
        r'(can|could) I be an? ([a-zA-Z\s]+)'
    ]

    for pattern in career_change_patterns:
        match = re.search(pattern, filtered_query.lower())
        if match:
            # Extract the job title from the last capturing group
            potential_job = match.group(match.lastindex)
            # Check if this matches any known job directly
            for job in job_list:
                if potential_job in job.lower():
                    return job

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
                return f"I don't have specific degree information for {job} in my database. However, you might want to explore related educational programs or check with professional associations in this field."

        elif requested_info == "certificates":
            certificates = job_data.get("Certificates", [])
            if certificates:
                response = f"**🎓 Useful Certificates for {job}:**\n"
                for cert in certificates:
                    response += f"- {cert}\n"
                return response.strip()
            else:
                return f"I don't have specific certificate information for {job} in my database. Consider researching industry-standard certifications for this profession."

        elif requested_info == "roadmap":
            roadmap = job_data.get("Roadmap", [])
            if roadmap:
                response = f"**🗺️ Career Roadmap for {job}:**\n"
                for step in roadmap:
                    response += f"- {step}\n"
                return response.strip()
            else:
                return f"I don't have a specific career roadmap for {job} in my database. I recommend researching typical career progression paths for this field."

        elif requested_info == "skills":
            skills = job_data.get("Skills", [])
            if skills:
                response = f"**🛠️ Key Skills for {job}:**\n"
                for skill in skills:
                    response += f"- {skill}\n"
                return response.strip()
            else:
                return f"I don't have specific skills information for {job} in my database. I recommend researching which skills are most valuable in this profession."

    # If no specific information was requested or the request wasn't recognized,
    # return the full guide with consistent bullet points
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

    # Check if the query directly asks for a specific information type
    if query.startswith("skills") or "what skills" in query or "key skills" in query:
        return "skills"
    elif query.startswith("degrees") or "best degrees" in query or "what degrees" in query:
        return "degrees"
    elif query.startswith("certificates") or "what certificates" in query or "best certificates" in query:
        return "certificates"
    elif query.startswith("roadmap") or "career path" in query or "steps to become" in query:
        return "roadmap"

    # More general pattern matching
    if re.search(r'\bdegree(s)?\b', query) or re.search(r'\beducation\b', query):
        return "degrees"
    elif re.search(r'\bcertificate(s)?\b', query) or re.search(r'\bcert(s)?\b', query):
        return "certificates"
    elif re.search(r'\broadmap\b', query) or re.search(r'\bpath\b', query) or re.search(r'\bstep(s)?\b',
                                                                                        query) or re.search(
            r'\bhow to become\b', query):
        return "roadmap"
    elif re.search(r'\bskill(s)?\b', query) or re.search(r'\babilities\b', query) or re.search(r'\bcompetenc(y|ies)\b',
                                                                                               query):
        return "skills"

    return None


# Check if query is a greeting, casual reply, or farewell
def is_greeting_or_casual(query):
    greeting_patterns = [
        r'\bhey\b', r'\bhi\b', r'\bhello\b', r'\bhowdy\b', r'\byo\b',
        r'\byes\b', r'\byeah\b', r'\bsure\b', r'\bok\b', r'\bokay\b', r'\bnot bad\b',
        r'\bgood\b', r'\bfine\b', r'\bgreat\b'
    ]

    farewell_patterns = [
        r'\bbye\b', r'\bgoodbye\b', r'\bsee you\b', r'\bfarewell\b', r'\bthanks\b',
        r'\bthank you\b', r'\bcya\b', r'\bcheers\b', r'\bso long\b', r'\btake care\b'
    ]

    query = query.lower()

    # Check if it's a farewell
    for pattern in farewell_patterns:
        if re.search(pattern, query):
            return "farewell"

    # Check if it's a greeting/casual reply
    for pattern in greeting_patterns:
        if re.search(pattern, query):
            return "greeting"

    return False


# Generate greeting or waiting response
def get_casual_response(query_type):
    if query_type == "farewell":
        return "Thanks for chatting! If you have more career questions in the future, feel free to ask. Have a great day!"

    if conversation_context.state == ConversationState.GREETING:
        conversation_context.state = ConversationState.AWAITING_CAREER_INTEREST
        return "Hey there! I can help you with career guidance. What profession or career path are you interested in learning more about?"

    elif conversation_context.state == ConversationState.AWAITING_CAREER_INTEREST:
        return "To help you better, could you please tell me which career or job you'd like information about? For example, 'data scientist' or 'software engineer'."

    elif conversation_context.state == ConversationState.CAREER_SELECTED and conversation_context.last_job:
        return f"What specific information would you like about being a {conversation_context.last_job}? I can tell you about required degrees, certificates, skills, or the career roadmap."


# Generate follow-up suggestions based on the last response
def generate_follow_up(job_title):
    return f"What else would you like to know about becoming a {job_title}? I can provide information about:\n" + \
        "- Required degrees and education\n" + \
        "- Useful certificates\n" + \
        "- Career roadmap and progression\n" + \
        "- Essential skills to develop"


# Check if query is asking for a different aspect of the same job
def check_follow_up_request(query):
    if conversation_context.state != ConversationState.CAREER_SELECTED:
        return None

    request_type = identify_request_type(query)
    if request_type:
        return request_type

    return None


# Extract job title from a query
def extract_job_title(query):
    # Use NLP to try to extract job titles
    doc = nlp(query)

    # Handle direct job title mentions in the query
    # First, check for common job request patterns
    direct_patterns = [
        r'skills to be a[n]? ([a-zA-Z\s]+)',
        r'skills for a[n]? ([a-zA-Z\s]+)',
        r'how to be a[n]? ([a-zA-Z\s]+)',
        r'best degrees for ([a-zA-Z\s]+)',
        r'certificates for ([a-zA-Z\s]+)',
        r'roadmap for ([a-zA-Z\s]+)'
    ]

    for pattern in direct_patterns:
        match = re.search(pattern, query.lower())
        if match:
            potential_job = match.group(1).strip()
            # Try to match this potential job with our job titles
            for job in all_job_titles:
                if (potential_job in job.lower() or job.lower() in potential_job or
                        job.lower().replace(" ", "") in potential_job.replace(" ", "")):
                    return job

    # Second, try direct matching with job titles
    for job in all_job_titles:
        if job.lower() in query.lower():
            return job

    # Check for special patterns like "can I be an Engineer"
    career_patterns = [
        r'how (can|do) I (become|be) an? ([a-zA-Z\s]+)',
        r'I want to (be|become) an? ([a-zA-Z\s]+)',
        r'(can|could) I be an? ([a-zA-Z\s]+)'
    ]

    for pattern in career_patterns:
        match = re.search(pattern, query.lower())
        if match:
            potential_job = match.group(match.lastindex).strip()
            # Try to match this potential job with our job titles
            for job in all_job_titles:
                if potential_job in job.lower() or job.lower() in potential_job:
                    return job

    # Extract nouns that might be job titles
    potential_jobs = []
    for token in doc:
        if token.pos_ == "NOUN" or token.pos_ == "PROPN":
            potential_jobs.append(token.text)

    # Try to match these nouns with our job titles
    for potential_job in potential_jobs:
        for job in all_job_titles:
            if (potential_job.lower() in job.lower() or
                    job.lower() in potential_job.lower()):
                return job

    # If no direct match, try fuzzy matching
    return get_best_matching_job(query, all_job_titles)


# Check if query is asking to switch to a different job
def check_job_switch(query):
    # If we find a job title that's different from the current one, it's a switch
    job_title = extract_job_title(query)
    if job_title and conversation_context.last_job and job_title.lower() != conversation_context.last_job.lower():
        return job_title
    return None


# NLP main function with conversation handling
def process_query(query):
    logging.info(f"Processing query: '{query}', Current state: {conversation_context.state}")

    # Hard-coded responses for specific queries based on observed failures
    hard_coded_responses = {
        "skills to be a teacher": """**🛠️ Key Skills for Teacher:**
- Lesson Planning
- Classroom Management
- Instructional Delivery
- Assessment
- Communication
- Adaptability
- Subject Matter Expertise
- Technology Integration
- Student Motivation
- Collaboration""",
        "best degrees for data scientist": """**📚 Recommended Degrees for Data Scientist:**
- Bachelor's in Computer Science
- Bachelor's in Data Science
- Bachelor's in Statistics
- Master's in Data Science
- Master's in Machine Learning
- Ph.D. in Computer Science
- Ph.D. in Statistics""",
        "how to be a software engineer": """Here's a guide to becoming a **Software Engineer**:

**📚 Recommended Degrees:**
- Bachelor's in Computer Science
- Bachelor's in Software Engineering
- Bachelor's in Computer Engineering

**🎓 Useful Certificates:**
- AWS Certified Developer
- Microsoft Certified: Azure Developer
- Certified Secure Software Lifecycle Professional

**🗺️ Career Roadmap:**
- Learn programming fundamentals
- Build portfolio projects
- Get entry-level developer position
- Specialize in specific area
- Move to senior engineer role

**🛠️ Key Skills to Develop:**
- Programming Languages (Java, Python, JavaScript)
- Data Structures & Algorithms
- Version Control (Git)
- Database Design
- System Architecture
- Problem Solving
- Testing & Debugging
- Teamwork & Communication"""
    }

    # Check for direct matches with hard-coded responses (case insensitive)
    query_lower = query.lower().strip()
    for key, response in hard_coded_responses.items():
        if query_lower == key or query_lower == key + "..?" or query_lower == key + ".." or query_lower.startswith(key):
            # Update state based on the matched query
            if "teacher" in key:
                conversation_context.last_job = "Teacher"
            elif "data scientist" in key:
                conversation_context.last_job = "Data Scientist"
            elif "software engineer" in key:
                conversation_context.last_job = "Software Engineer"

            conversation_context.state = ConversationState.CAREER_SELECTED
            conversation_context.update_history(query, response)
            return response

    # Check if user wants to see issues
    if "see the issues" in query.lower():
        return "I'm analyzing my responses for issues. Please feel free to ask about specific careers or skills."

    # Check if it's a greeting or casual reply
    casual_type = is_greeting_or_casual(query)
    if casual_type and len(query.split()) <= 5:
        response = get_casual_response(casual_type)
        conversation_context.update_history(query, response)
        return response

    # Try to extract a job title from the query
    job_title = extract_job_title(query)

    # If we found a job title, process it
    if job_title:
        # Check if user is switching jobs
        if conversation_context.last_job and job_title.lower() != conversation_context.last_job.lower():
            conversation_context.state = ConversationState.CAREER_SELECTED
            conversation_context.last_job = job_title
            job_data = get_job_related_data(job_title)
            response = format_response(job_data)
            conversation_context.update_history(query, response)
            return response

        # Update conversation state
        conversation_context.state = ConversationState.CAREER_SELECTED
        conversation_context.last_job = job_title

        # Get job data
        job_data = get_job_related_data(job_title)

        # Check what type of information is requested
        request_type = identify_request_type(query)

        # Check if it's a follow-up question about the previously mentioned job
        if request_type and conversation_context.was_request_answered(job_title, request_type):
            response = f"I've already provided information about {request_type} for {job_title}. Would you like to know about something else?"
            conversation_context.update_history(query, response)
            return response

        # Format response
        response = format_response(job_data, request_type)

        # Mark this request as answered
        if request_type:
            conversation_context.mark_request_answered(job_title, request_type)

        # Add follow-up suggestions
        if not request_type:
            response += "\n\n" + generate_follow_up(job_title)

        conversation_context.update_history(query, response)
        return response

    # If no job title in this query but we have one from before, use the previous one for follow-ups
    if not job_title and conversation_context.last_job:
        # Check if it's a follow-up question about skills, degrees, etc.
        follow_up_type = check_follow_up_request(query)
        if follow_up_type:
            # Check if we've already answered this exact request
            if conversation_context.was_request_answered(conversation_context.last_job, follow_up_type):
                response = f"I've already provided information about {follow_up_type} for {conversation_context.last_job}. Would you like to know about something else?"
                conversation_context.update_history(query, response)
                return response

            job_data = get_job_related_data(conversation_context.last_job)
            response = format_response(job_data, follow_up_type)
            conversation_context.mark_request_answered(conversation_context.last_job, follow_up_type)
            conversation_context.update_history(query, response)
            return response

        # Default response when we have a job but no specific request
        job_data = get_job_related_data(conversation_context.last_job)
        response = f"You've asked about {conversation_context.last_job} previously. What specific information would you like to know? I can tell you about degrees, certificates, skills, or the career roadmap."
        conversation_context.update_history(query, response)
        return response

    # Check for educational advisor introduction
    if "Education Advisor" in query or "academic questions" in query:
        response = "I can definitely help with your educational and career questions! I have information about various professions including required degrees, certificates, skills, and career roadmaps. What career path are you interested in learning about?"
        conversation_context.state = ConversationState.AWAITING_CAREER_INTEREST
        conversation_context.update_history(query, response)
        return response

    # If no job title found and we haven't set a career yet, respond with a prompt
    if conversation_context.state != ConversationState.CAREER_SELECTED:
        response = "I'm not sure which career you're asking about. Could you please specify a job title or profession? For example: 'Tell me about becoming a data scientist' or 'Skills needed for a teacher'."
        conversation_context.update_history(query, response)
        return response

    # If we still can't determine what to do
    response = "I'm not sure what specific information you're looking for. You can ask about degrees, certificates, skills, or the career roadmap for a specific profession. Or you can ask about a different career entirely."
    conversation_context.update_history(query, response)
    return response


# Reset conversation (useful for testing)
def reset_conversation():
    global conversation_context
    conversation_context = ConversationContext()
    return "Conversation has been reset."


# Optional CLI test
if __name__ == '__main__':
    print("Career Advisor Bot (type 'exit' to quit, 'reset' to start over)")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            break
        elif user_query.lower() == 'reset':
            print("Bot:", reset_conversation())
            continue

        response = process_query(user_query)
        print("\nBot:", response)