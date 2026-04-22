# LLM-P1-16-Group-Project

# Food Chatbot

The Food Chatbot is an intelligent, highly personalised Singaporean food recommendation chatbot. Designed with a localised Singlish persona, it leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to provide hyper-specific dining suggestions based on user dietary restrictions, location, budget and real-time live web data.

## How the Architecture Works
1. **Intent Classification:** When a user sends a message, chatbot_main.py uses an LLM to decide if the user wants to search for food, save a preference, ask a live question or just chat
2. **Dual-Database Retrieval:** If searching, the app queries both a Primary and Secondary pgvector database to pull 80 candidate restaurants
3. **Re-ranking & Filtering:** A custom scoring algorithm evaluates the 80 candidates against the user's saved JSONB preferences in PostgreSQL. It heavily penalises mismatched locations or dietary constraints
4. **The 2/2 Split:** The system picks the top 2 matches from the Primary database and the top 2 from the Secondary database to guarantee quality and variety
5. **Streaming Generation:** The selected context is passed to GPT-4o which streams its response back to the user via Flask's stream_with_context

## Key Features
* **Smart Intent Routing:** Automatically detects whether a user is looking for recommendations, saving preferences or asking for live operational info
* **Hybrid Search & Reranking:** Uses `pgvector` for semantic search, layered with a custom lexical reranker that strictly enforces location bounds, dietary needs (e.g., Halal, Vegan) and prioritises highly-rated restaurants
* **Streaming Responses:** Provides a ChatGPT-like real-time typing experience using Flask streaming and Server-Sent Events (SSE)
* **Live Web Search:** Integrates with DuckDuckGo to pull real-time operational details (opening hours, contact info) when the database lacks them
* **User Accounts & Personalisation:** Secure authentication flow allowing users to save their allergies, favorite cuisines, spice tolerance and budgets which dynamically influence all future recommendations
* **Reflection Agent:** An internal AI critic evaluates search results before presenting them to the user which automatically revises queries if the initial results don't match the user's constraints

## Tech Stack
* **Backend:** Python 3, Flask, Werkzeug
* **AI & Orchestration:** OpenAI API (`gpt-4o`, `gpt-3.5-turbo`), LangChain, OpenAI Embeddings (`text-embedding-3-small`)
* **Database:** PostgreSQL with `pgvector` extension, `psycopg2`
* **Frontend:** Vanilla HTML/CSS/JS, marked.js for Markdown parsing

---

## Installation & Setup

### 1. Prerequisites
* **Python 3.9+** installed on your machine
* **PostgreSQL** installed and running
* **pgvector** extension installed on your PostgreSQL database

### 2. Clone the Repository
```bash
git clone <https://github.com/akram-25/LLM-P1-16-Group-Project>
cd LLM-P1-16-Group-Project
```

### 3. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 4. Configuration (API Keys & Environment Variables)
The application relies on an environment file to securely load API keys and database credentials

1. Create a new file named apikeys.env in the root directory of the project (the same folder as app.py)
2. Add the following variables to the file and replace the placeholder values with the one provided in the G16keys.txt file:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=your_database_name
POSTGRES_USER=your_postgres_username
POSTGRES_PASSWORD=your_postgres_password
```

### 5. Database Setup
**Please email us before accessing the application so that we can:**
1. Activate the PostgreSQL database to run this project.
    - The application expects the both vector store collections to exist and be populated with restaurant embeddings
2. Whitelist your IP address
    - As the school's IP addresses do not work, please use your mobile's hotspot and provide the IP address for us to whitelist

The application will automatically verify and initialise missing columns in the user authentication tables upon startup via db.init_db().

## Running the Application
Once your database is running and your apikeys.env file is configured, you can start the Flask development server:
python app.py

The application will be available in your browser at http://127.0.0.1:5000/

## Project Structure
- **app.py:** The main Flask server application handling routing, authentication and the streaming chat endpoint
- **chatbot_main.py:** The core LLM logic, containing the intent router, vector database search, custom reranking algorithm, reflection loop and response generation
- **db.py:** Handles all direct PostgreSQL database operations, including user authentication, chat history persistence and JSONB preference updates
- **prompts.py:** Contains the system instructions and behavioral prompts for the LLM (Intent Router, Persona and Critic)
- **templates/:** Contains the frontend HTML files (index.html, login.html and settings.html)

```text
├── app.py                  # Main Flask application, routing and streaming logic
├── chatbot_main.py         # AI engine: Intent routing, RAG pipeline and LLM generation
├── db.py                   # PostgreSQL database connections, auth and user preferences
├── prompts.py              # System prompts for Intent Classification, Persona and Critic
├── apikeys.env             # Environment variables (API keys, DB credentials)
├── food_places/
│   ├── food_places_primary.csv   # Highly curated list of restaurants
│   └── food_places_secondary.csv # Extended database of food places
└── templates/
    ├── index.html          # Main chat interface
    ├── login.html          # User authentication page
    └── settings.html       # Preference management UI
```

## Team Members (Group 16)
* Chan Kai Wen
* Mohaamed Akram
* Mohamed Thabith
* Song Jie Min Jaymee
* Wong Zi Qin