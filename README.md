# Education Chatbot Backend

This is a backend service for an Education Chatbot that helps users with career guidance, providing information about educational requirements, skills, certificates, and career roadmaps.

## Setup Instructions

### 1. Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### 2. Installation

Run the setup script to install all dependencies:

```bash
python setup.py
```

The setup script will:
- Install required Python packages
- Install the spaCy language model (en_core_web_sm)
- Set up the Dataset directory

If the setup script encounters any issues with installing the spaCy model, you can install it manually:

```bash
python -m spacy download en_core_web_sm
```

### 3. Data Files

Place the following data files in the `Dataset` directory:
- `Certificates.xlsx`
- `Degrees.xlsx`
- `Roadmaps.xlsx`
- `Skills.json`
- `greeting.json` 

### 4. Running the Application

Start the server:

```bash
python app.py
```

The service will be available at http://localhost:5000

## API Endpoints

- `GET /`: Check if the service is running
- `POST /chat`: Send a user message and get a response
- `GET /history`: Get conversation history
- `POST /clear_history`: Clear conversation history
- `POST /feedback`: Submit feedback about a response
- `POST /learn`: Add new information to the knowledge base
- `GET /knowledge`: Get stored knowledge for a specific job

## Learning Capabilities

The chatbot has machine learning capabilities that allow it to:
1. Learn from user feedback
2. Store corrections and new information
3. Improve responses over time
4. Apply learned knowledge to future responses
5. Track which responses are helpful

## Troubleshooting

If you encounter any issues with spaCy, ensure:
1. You have installed the model with `python -m spacy download en_core_web_sm`
2. Your Python environment has sufficient permissions to install packages
3. You are using the same Python environment for both setup and running the application


