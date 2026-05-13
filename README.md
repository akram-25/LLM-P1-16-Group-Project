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

# FoodKakiBot — Google Cloud PostgreSQL (pgvector) Setup Guide

This guide explains the process on how to create a fresh PostgreSQL instance on Google Cloud SQL, enabling the `pgvector` extension and wiring it into the FoodKaki chatbot.

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Run the app and population scripts |
| [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) | Manage Cloud SQL from the terminal |
| A Google Cloud project with billing enabled | Host the database |
| An OpenAI API key | Embeddings + LLM calls |

Install Python dependencies once you have the repo:

```bash
pip install -r requirements.txt
```

---

## Step 1 — Create a Cloud SQL (PostgreSQL) Instance

### Option A: Google Cloud Console (GUI)

1. Go to **Cloud SQL** in the [Google Cloud Console](https://console.cloud.google.com/sql)
2. Click **Create Instance** → choose **PostgreSQL**
3. Set the following:
   - **Database version**: PostgreSQL 15 or newer
   - **Instance ID**: e.g. `foodkaki-db`
   - **Password**: set a strong password for the `postgres` user — save it
   - **Region**: choose one close to you (e.g. `asia-southeast1` for Singapore)
   - **Machine type**: `db-f1-micro` is sufficient for development. If using free google credits, might not be able to choose the cheapest option
4. Under **Connections**, enable **Public IP** (you can restrict by authorised network later)
5. Click **Create Instance** and wait (~3–5 minutes)

### Option B: gcloud CLI
```bash
gcloud sql instances create foodkaki-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=asia-southeast1 \
  --root-password=YOUR_ROOT_PASSWORD
```

---

## Step 2 — Create the Application Database

Replace `foodkaki-db` with your instance ID if different

```bash
gcloud sql databases create foodkaki \
  --instance=foodkaki-db
```

This creates a database named `foodkaki` inside the instance

---

## Step 3 — Create a Database User

Avoid using the root `postgres` account for the app

```bash
gcloud sql users create foodkaki_user \
  --instance=foodkaki-db \
  --password=YOUR_APP_PASSWORD
```

Grant the user access to the database by connecting as `postgres` (see Step 5) and running:

```sql
GRANT ALL PRIVILEGES ON DATABASE foodkaki TO foodkaki_user;
```

---

## Step 4 — Enable the pgvector Extension

Connect to the instance (see Step 5) and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

> **Important**: This must be done once on the `foodkaki` database before running any population scripts.  
> Cloud SQL supports `pgvector` natively from PostgreSQL 15+. If you see `ERROR: extension "vector" does not exist`, verify your PostgreSQL version is 15 or higher.

---

## Step 5 — Allow Your IP to Connect

Cloud SQL blocks all connections by default

1. Find your public IP:
   ```bash
   curl ifconfig.me
   ```

2. Authorize it:
   ```bash
   gcloud sql instances patch foodkaki-db \
     --authorized-networks=YOUR_IP/32
   ```

   Or in the Console: go to your instance → **Connections** → **Networking** → **Add a network**

3. Get the instance's public IP:
   ```bash
   gcloud sql instances describe foodkaki-db --format="value(ipAddresses[0].ipAddress)"
   ```
   Save this — it goes in `apikeys.env` as `POSTGRES_HOST`

### Connecting via psql (to run SQL commands)

```bash
psql -h YOUR_INSTANCE_IP -U foodkaki_user -d foodkaki
```

---

## Step 6 — Create `apikeys.env`

Create a file named `apikeys.env` in the **project root** (same folder as `app.py`):

```
OPENAI_API_KEY=sk-...your-openai-key...

POSTGRES_HOST=YOUR_INSTANCE_PUBLIC_IP
POSTGRES_PORT=5432
POSTGRES_DB=foodkaki
POSTGRES_USER=foodkaki_user
POSTGRES_PASSWORD=YOUR_APP_PASSWORD
```

> Do **not** commit `apikeys.env` to git. It is already in `.gitignore`

---

## Step 7 — Populate the Vector Databases

The app uses two pgvector collections loaded from CSV files in `food_places/`. Run these **once** after the database is ready:

```bash
python postgres_scripts/create_primary_postgres.py
python postgres_scripts/create_secondary_postgres.py
```

Each script:
- Reads the CSV (`food_places_primary.csv` / `food_places_secondary.csv`)
- Generates OpenAI embeddings for each restaurant record (costs a small amount of API credits)
- Uploads to the pgvector collections `foodkaki_restaurants` and `foodkaki_restaurants_secondary`

This takes a few minutes depending on dataset size and network speed. Progress is printed batch by batch.

> Re-running either script **wipes and repopulates** that collection (`pre_delete_collection=True`)

---

## Step 8 — Run the App

```bash
python app.py
```

On startup, `app.py` automatically calls `db.init_db()`, which creates the application tables (`users`, `chat_history`, `user_preferences`, `search_history`, `user_favorites`) if they do not exist. No manual SQL is needed for these.

Visit `http://127.0.0.1:5000/` in your browser

---

## Verifying the Setup

Connect to the database and check:

```sql
-- Check pgvector is installed
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check vector collections exist (created by langchain-postgres)
SELECT collection_id, name FROM langchain_pg_collection;

-- Check app tables exist
\dt
```

You should see both `foodkaki_restaurants` and `foodkaki_restaurants_secondary` in `langchain_pg_collection`, plus the five app tables

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `connection refused` | IP not whitelisted | Add your IP in Cloud SQL → Connections |
| `extension "vector" does not exist` | pgvector not enabled | Run `CREATE EXTENSION IF NOT EXISTS vector;` as a superuser |
| `could not connect to server` | Wrong host/port in `apikeys.env` | Re-check `POSTGRES_HOST` is the public IP, port is `5432` |
| Population script hangs | OpenAI rate limit or network timeout | Re-run as the script uses `pre_delete_collection=True` so it is safe to retry |
| `password authentication failed` | Wrong credentials | Double-check `POSTGRES_USER` / `POSTGRES_PASSWORD` in `apikeys.env` |
| App starts but search returns nothing | Vector collections empty | Run both `create_primary_postgres.py` and `create_secondary_postgres.py` |

---

## Cost Notes

- **Cloud SQL db-f1-micro**: ~USD $7–10/month when running continuously; free tier may apply
- **OpenAI embeddings** (`text-embedding-3-small`): very cheap — populating both datasets costs well under USD $1
- Stop the Cloud SQL instance when not in use to avoid charges:
  ```bash
  gcloud sql instances patch foodkaki-db --activation-policy=NEVER
  ```
  Restart it with `--activation-policy=ALWAYS`

## Team Members (ITP Group 8)
* Chan Kai Wen
* Mohaamed Akram
* Mohamed Thabith
* Song Jie Min Jaymee
* Wong Zi Qin