import os
import json
from dotenv import load_dotenv

from openai import OpenAI
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings

# ==========================================
# 1. SETUP & CONNECTION
# ==========================================
load_dotenv('apikeys.env')

CHROMA_SERVER_URL = os.getenv('CHROMA_SERVER_URL')
CHROMA_API_TOKEN = os.getenv('CHROMA_API_TOKEN')

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

print("🔗 Connecting to Cloud Memory...")

chroma_auth_client = None
try:
    chroma_auth_client = chromadb.HttpClient(
        host=CHROMA_SERVER_URL,
        port=443,
        ssl=True,
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=CHROMA_API_TOKEN,
            anonymized_telemetry=False
        )
    )
    chroma_auth_client.heartbeat()
    print("   ✅ Server reachable.")
except Exception as e:
    print(f"\n🛑 WARNING: Could not connect to Database. {e}")

embedding_fn = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# NOTE: We keep "foodkaki_restaurants" here so it still connects to the existing DB
vector_db = Chroma(
    client=chroma_auth_client,
    collection_name="foodkaki_restaurants",
    embedding_function=embedding_fn
) if chroma_auth_client else None


# ==========================================
# 2. USER MEMORY (Long-Term Storage)
# ==========================================
USER_DB_FILE = "users.json"

def load_user_profile(username):
    try:
        with open(USER_DB_FILE, 'r') as f:
            data = json.load(f)
        return data.get(username, {}) 
    except FileNotFoundError:
        return {}

def save_user_preference(username, key, value):
    try:
        with open(USER_DB_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    
    if username not in data:
        data[username] = {}
        
    if key not in data[username]:
        data[username][key] = []
        
    if isinstance(data[username][key], str):
        data[username][key] = [data[username][key]]
        
    current_values = [v.lower() for v in data[username][key]]
    
    if value.lower() not in current_values:
        data[username][key].append(value)
        print(f"   [Memory] Added to {username}'s {key}: {value}")
    
    with open(USER_DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)


# ==========================================
# 3. THE BRAIN (System Prompts & Router)
# ==========================================
INTENT_PROMPT = """
You are the "Router" for a Singapore food chatbot.
Classify the user's input into one of these intents:

1. SEARCH: User asks for food recommendations, details, location, prices, or "vibe".
   - Return: {"intent": "SEARCH", "keywords": "search terms"}

2. SAVE_PREF: User states a personal fact or identity.
   - **CRITICAL**: You must map the 'key' to one of these STANDARD keys: "name", "diet", "allergy", "location".
   - Return: {"intent": "SAVE_PREF", "key": "STANDARD_KEY", "value": "extracted value"}

3. CHAT: Social greetings, small talk, OR questions about the user's own history/profile.
   - Return: {"intent": "CHAT"}

4. BLOCK: Topics CLEARLY NOT related to food/dining/profile (e.g., sports, coding, politics).
   - Return: {"intent": "BLOCK"}

Reply ONLY JSON.
"""

PERSONA_PROMPT = """
You are a helpful Singaporean food chatbot.
- Speak in natural Singlish (use "lah", "lor", "shiok", "paiseh", "can").
- Be friendly but concise.
- Use the provided Context (Database Results) to recommend places enthusiastically.
- Use the User History to remember what we just talked about.
"""

GRADER_PROMPT = """
You are a Relevance Grader for a restaurant search engine.
User Query: "{user_query}"

Retrieved Database Results:
{context_str}

Task: Evaluate if the retrieved results accurately satisfy the user's query (pay special attention to cuisines, location, and "vibe").
- If the results are good, return: {{"relevant": true, "improved_query": ""}}
- If the results DO NOT match (or are empty), return: {{"relevant": false, "improved_query": "new, better search keywords"}}

Reply ONLY in JSON.
"""

def get_intent(user_query):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=0,
            response_format={"type": "json_object"} 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"   [Router Error] {e}")
        return {"intent": "CHAT"}


# ==========================================
# 4. CORE FUNCTIONS (RAG, Reflection & Generation)
# ==========================================
def search_cloud_db(query_text):
    if not vector_db:
        return []
    try:
        results = vector_db.similarity_search(query_text, k=4)
        clean_results = []
        for doc in results:
            info = {
                "name": doc.metadata.get('name', 'Unknown Place'),
                "category": doc.metadata.get('category', 'Food'),
                "description": doc.page_content
            }
            clean_results.append(info)
        return clean_results
    except Exception as e:
        print(f"   ❌ DB Search Failed: {e}")
        return []

def reflective_search(user_input, initial_keywords):
    print(f"   🔎 [Agent] Searching Cloud DB for: '{initial_keywords}'...")
    results = search_cloud_db(initial_keywords)
    
    context_str = ""
    if results:
        for r in results:
            context_str += f"- {r['name']} ({r['category']}): {r['description']}\n"
    else:
        context_str = "No results found."

    print("   🤔 [Reflection] Grading the search results...")
    grader_msg = GRADER_PROMPT.format(user_query=user_input, context_str=context_str)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "system", "content": grader_msg}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        grade = json.loads(response.choices[0].message.content)
        
        if grade.get("relevant") == True and results:
            print("   ✅ [Reflection] Results look good. Proceeding.")
            return results
        else:
            new_query = grade.get("improved_query", initial_keywords)
            print(f"   🔄 [Reflection] Results poor. Retrying DB search with: '{new_query}'")
            return search_cloud_db(new_query)
            
    except Exception as e:
        print(f"   ⚠️ [Grader Error] {e}. Falling back to first attempt.")
        return results

def generate_response_with_history(new_user_input, chat_history, context_data=None):
    messages = [{"role": "system", "content": PERSONA_PROMPT}]
    
    user_profile = load_user_profile("Akram") 
    if user_profile:
        prefs_str = f"IMPORTANT - User Preferences: {json.dumps(user_profile)}"
        messages.append({"role": "system", "content": prefs_str})

    if context_data:
        context_str = "Database Results:\n"
        for place in context_data:
            context_str += f"- {place['name']} ({place['category']}): {place['description']}\n"
        messages.append({"role": "system", "content": context_str})
    
    messages.extend(chat_history)
    messages.append({"role": "user", "content": new_user_input})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7 
    )
    return response.choices[0].message.content