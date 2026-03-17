import os
import json
from dotenv import load_dotenv

from openai import OpenAI
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings

from langchain_community.tools import DuckDuckGoSearchRun

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

primary_vector_db = Chroma(
    client=chroma_auth_client,
    collection_name="foodkaki_restaurants",
    embedding_function=embedding_fn
) if chroma_auth_client else None

secondary_vector_db = Chroma(
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
Classify the user's input into one of these intents based on the User Query and the Recent Chat History.

1. SEARCH: User asks for food recommendations, details, location, prices, or "vibe".
   - **CRITICAL**: Resolve pronouns using history.
   - Return: {"intent": "SEARCH", "keywords": "search terms"}

2. LIVE_SEARCH: User asks for operational details like opening hours, phone numbers, or exact addresses.
   - **CRITICAL**: Resolve pronouns using history. Always include the restaurant name and the specific detail requested in the query.
   - Example: {"intent": "LIVE_SEARCH", "query": "Ah Seng Durian opening hours and phone number Singapore"}
   - Return: {"intent": "LIVE_SEARCH", "query": "exact search terms"}

3. SAVE_PREF: User states a personal fact or identity.
   - Return: {"intent": "SAVE_PREF", "key": "STANDARD_KEY", "value": "extracted value"}

4. SAVE_FAVORITE: User asks to save or bookmark a specific restaurant.
   - Return: {"intent": "SAVE_FAVORITE", "restaurant_name": "Extracted Name"}

5. CHAT: Social greetings, small talk, OR questions about the user's own history/profile.
   - Return: {"intent": "CHAT"}

6. BLOCK: Topics CLEARLY NOT related to food/dining/profile.
   - Return: {"intent": "BLOCK"}

Reply ONLY JSON.
"""

PERSONA_PROMPT = """
You are a helpful Singaporean food chatbot.
- Speak in natural Singlish (use "lah", "lor", "shiok", "paiseh", "can").
- Be friendly but concise.
- Use the User History to remember what we just talked about.

CRITICAL RULES FOR ANSWERING (DO NOT BREAK THESE):
1. FOR RECOMMENDATIONS: STRICTLY base your restaurant suggestions ONLY on the provided "Database Results" (Primary or Extended Selection). 
2. FOR OPERATIONAL INFO (Hours, Phone, Address): If the context includes "Live Web Search", you are AUTHORIZED to use that information to answer the user's specific question. Say something like, "I just checked online for you..."
3. NEVER invent or hallucinate information. If the Database Results or Web Search Results are empty, you MUST apologize and say you don't know.
4. When giving recommendations, present "Primary Selection" places as "Top Picks 🌟", and "Extended Selection" places as "More Options 🍽️".
"""

def get_intent(user_query, chat_history=[]):
    try:
        # Convert the last 4 messages of history into a readable string for the router
        history_text = "None"
        if chat_history:
            # Grab just the last few turns so we don't waste tokens
            recent_context = chat_history[-4:] 
            history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_context])

        # Inject the history into the system prompt
        dynamic_system_prompt = INTENT_PROMPT + f"\n\n--- RECENT CHAT HISTORY ---\n{history_text}"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": dynamic_system_prompt},
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
    if not primary_vector_db or not secondary_vector_db:
        print("   ❌ Databases not fully initialized.")
        return []
        
    strict_filters = []
    
    # ONLY apply strict DB filters for dealbreakers (Health/Safety)
    if user_profile:
        if user_profile.get("diet") and len(user_profile["diet"]) > 0:
            strict_filters.append({"diet": {"$eq": user_profile["diet"][0].lower()}})
            
        if user_profile.get("allergy") and len(user_profile["allergy"]) > 0:
            for allergen in user_profile["allergy"]:
                strict_filters.append({"allergens": {"$ne": allergen.lower()}})

    search_filter = None
    if len(strict_filters) == 1:
        search_filter = strict_filters[0]
    elif len(strict_filters) > 1:
        search_filter = {"$and": strict_filters}

    try:
        # STAGE 1: Broad Vector Search (Fetch k=40 to cast a massive net)
        if search_filter:
            print(f"   ⚙️ [Filter] Applying strict Dealbreaker rules: {search_filter}")
            primary_docs = primary_vector_db.similarity_search(query_text, k=40, filter=search_filter)
            secondary_docs = secondary_vector_db.similarity_search(query_text, k=40, filter=search_filter)
        else:
            primary_docs = primary_vector_db.similarity_search(query_text, k=40)
            secondary_docs = secondary_vector_db.similarity_search(query_text, k=40)
            
        # Combine all 80 results into a single pool
        all_docs = []
        for doc in primary_docs:
            all_docs.append({"tier": "Primary Selection", "doc": doc})
        for doc in secondary_docs:
            all_docs.append({"tier": "Extended Selection", "doc": doc})

        # STAGE 2: Lightweight Lexical Reranker
        query_words = set(query_text.lower().split())
        scored_results = []
        
        for item in all_docs:
            doc = item["doc"]
            restaurant_name = doc.metadata.get('name', 'Unknown Place').lower()
            
            score = 0 
            
            # Massive boost if the exact prompt is in the name
            if query_text.lower() in restaurant_name:
                score += 500 
                
            # Strong boost for partial name matches
            for word in query_words:
                if len(word) > 3 and word in restaurant_name: 
                    score += 100
                    
            scored_results.append({
                "score": score,
                "name": doc.metadata.get('name', 'Unknown Place'),
                "category": doc.metadata.get('category', 'Food'),
                "tier": item["tier"],
                "description": doc.page_content
            })

        # Sort the results so the highest scores (exact matches) bubble to the top
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        # STAGE 3: Return the top 4 absolute best results to the LLM
        final_results = scored_results[:4]
        
        # Clean up the output so we don't send the raw 'score' integer to the LLM
        clean_results = [{"name": r["name"], "category": r["category"], "tier": r["tier"], "description": r["description"]} for r in final_results]
        
        print(f"   ✅ [Search] Reranked {len(all_docs)} docs. Top match score: {scored_results[0]['score'] if scored_results else 0}")
        return clean_results
        
    except Exception as e:
        print(f"   ❌ DB Search Failed: {e}")
        return []

def generate_response_with_history(new_user_input, chat_history, context_data=None, user_profile=None):
    messages = [{"role": "system", "content": PERSONA_PROMPT}]
    
    # Let the LLM see the user's saved preferences!
    if user_profile:
        messages.append({"role": "system", "content": f"User Preferences to keep in mind: {user_profile}"})
    
    if context_data:
        # Dynamically label the data so the LLM doesn't get confused
        if len(context_data) == 1 and context_data[0].get("name") == "Live Web Search":
            context_str = "Live Web Search Results (You may use this to answer the user):\n"
        else:
            context_str = "Database Results:\n"
            
        for place in context_data:
            context_str += f"- {place['name']} ({place['category']} - {place['tier']}): {place['description']}\n"
            
        messages.append({"role": "system", "content": context_str})
    
    messages.extend(chat_history)
    messages.append({"role": "user", "content": new_user_input})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7 
    )
    return response.choices[0].message.content

# Initialize free web search tool
web_search_tool = DuckDuckGoSearchRun()

def search_live_web(query):
    print(f"   🌐 [Agent] Searching the live web for: '{query}'...")
    try:
        # Ask DuckDuckGo for the top snippets
        return web_search_tool.run(query)
    except Exception as e:
        print(f"   ❌ Web Search Failed: {e}")
        return "Sorry, I couldn't find those details online right now."