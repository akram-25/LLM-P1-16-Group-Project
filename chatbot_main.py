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

vector_db = Chroma(
    client=chroma_auth_client,
    collection_name="foodkaki_restaurants",
    embedding_function=embedding_fn
) if chroma_auth_client else None

vector_db_secondary = Chroma(
    client=chroma_auth_client,
    collection_name="foodkaki_restaurants_secondary",
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

1. SEARCH: User asks for food recommendations, restaurant suggestions, "where to eat", food details, location of eateries, prices, or dining "vibe".
   - Return: {"intent": "SEARCH", "keywords": "search terms"}

2. SAVE_PREF: User STATES, SETS, or CHANGES a personal preference or fact about themselves.
   - Examples: "I'm allergic to peanuts", "I don't eat pork", "change my spice tolerance to high", "my budget is $10", "I prefer Japanese food", "my name is John", "I live near Jurong"
   - Map the 'key' to one of these STANDARD keys: "name", "diet", "allergy", "cuisine", "budget", "spice", "location"
   - Return: {"intent": "SAVE_PREF", "key": "STANDARD_KEY", "value": "extracted value"}

3. CHAT: Social greetings, small talk, thank you, general food chat, questions about the chatbot itself (e.g. "what can you do?", "who are you?"), OR when the user ASKS or QUERIES about their own preferences/profile/history.
   - IMPORTANT: If the user is ASKING about their preferences (e.g. "what are my allergies?", "what are my dietary requirements?", "tell me my preferences", "do I have food restrictions?", "remind me what I like"), this is CHAT, NOT SAVE_PREF.
   - Return: {"intent": "CHAT"}

4. BLOCK: Topics CLEARLY NOT related to food/dining/profile/the chatbot (e.g., sports, coding, politics, homework).
   - Return: {"intent": "BLOCK"}

CRITICAL RULES:
- ASKING about preferences = CHAT (user wants to know their saved info)
- STATING or CHANGING a preference = SAVE_PREF (user provides new info to save)
- If unsure between SAVE_PREF and CHAT, choose CHAT.

Reply ONLY JSON.
"""

PERSONA_PROMPT = """
You are a helpful Singaporean food chatbot.
- Speak in natural Singlish (use "lah", "lor", "shiok", "paiseh", "can").
- Be friendly but concise.
- Use the provided Context (Database Results) to recommend places enthusiastically.
- Use the User History to remember what we just talked about.
- You have access to the user's saved food preferences (if any). When the user asks about their preferences, dietary restrictions, allergies, or profile, ALWAYS refer to the saved preferences and list them clearly.
- When recommending food, take the user's preferences into account (e.g. avoid allergens, respect dietary restrictions, match budget and cuisine preferences).
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
def search_cloud_db(query_text, user_profile=None):
    if not vector_db and not vector_db_secondary:
        return []
        
    search_filter = {}
    
    if user_profile:
        if "diet" in user_profile and len(user_profile["diet"]) > 0:
            search_filter["diet"] = user_profile["diet"][0].lower()

    all_results = []
    seen_names = set()

    # Search both collections
    for db, db_name in [(vector_db, "primary"), (vector_db_secondary, "secondary")]:
        if not db:
            continue
        try:
            if search_filter:
                print(f"   ⚙️ [Filter] Applying strict rules on {db_name}: {search_filter}")
                results = db.similarity_search(query_text, k=4, filter=search_filter)
            else:
                results = db.similarity_search(query_text, k=4)

            for doc in results:
                name = doc.metadata.get('name', 'Unknown Place')
                if name not in seen_names:
                    seen_names.add(name)
                    all_results.append({
                        "name": name,
                        "category": doc.metadata.get('category', 'Food'),
                        "description": doc.page_content
                    })
        except Exception as e:
            print(f"   ❌ DB Search Failed ({db_name}): {e}")

    return all_results

def reflective_search(user_input, initial_keywords, user_profile=None):
    print(f"   🔎 [Agent] Searching Cloud DB for: '{initial_keywords}'...")
    results = search_cloud_db(initial_keywords, user_profile)
    
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
            return search_cloud_db(new_query, user_profile)
            
    except Exception as e:
        print(f"   ⚠️ [Grader Error] {e}. Falling back to first attempt.")
        return results

def generate_response_with_history(new_user_input, chat_history, context_data=None, user_profile=None):
    messages = [{"role": "system", "content": PERSONA_PROMPT}]
    
    if user_profile:
        # Build a human-readable preferences summary
        pref_lines = []
        if user_profile.get('allergy'):
            pref_lines.append(f"Food Allergies: {', '.join(user_profile['allergy'])}")
        if user_profile.get('diet'):
            pref_lines.append(f"Dietary Restrictions: {', '.join(user_profile['diet'])}")
        if user_profile.get('cuisine'):
            pref_lines.append(f"Favourite Cuisines: {', '.join(user_profile['cuisine'])}")
        if user_profile.get('budget'):
            pref_lines.append(f"Budget Range: {', '.join(user_profile['budget'])}")
        if user_profile.get('spice'):
            pref_lines.append(f"Spice Tolerance: {', '.join(user_profile['spice'])}")
        if user_profile.get('location'):
            pref_lines.append(f"Preferred Area: {', '.join(user_profile['location'])}")
        if user_profile.get('notes'):
            pref_lines.append(f"Additional Notes: {', '.join(user_profile['notes'])}")

        if pref_lines:
            prefs_str = "IMPORTANT — User's Saved Preferences:\n" + "\n".join(pref_lines)
            prefs_str += "\n\nIf the user asks about their preferences, repeat these back to them clearly."
        else:
            prefs_str = "The user has not set any food preferences yet. If they ask, let them know they can set preferences via the ⚙️ settings page."
        messages.append({"role": "system", "content": prefs_str})
    else:
        messages.append({"role": "system", "content": "The user has not set any food preferences yet. If they ask, let them know they can set preferences via the ⚙️ settings page."})

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