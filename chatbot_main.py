import os
import sys
import json
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.tools import DuckDuckGoSearchRun

# ==========================================
# 1. SETUP & CONNECTION
# ==========================================
load_dotenv('apikeys.env')

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

PG_CONNECTION = (
    f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

print("🔗 Connecting to PostgreSQL vector store...")

embedding_fn = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

try:
    primary_vector_db = PGVector(
        connection=PG_CONNECTION,
        collection_name="foodkaki_restaurants",
        embeddings=embedding_fn,
    )
    secondary_vector_db = PGVector(
        connection=PG_CONNECTION,
        collection_name="foodkaki_restaurants_secondary",
        embeddings=embedding_fn,
    )
    print("   ✅ Vector store connected.")
except Exception as e:
    print(f"\n🛑 WARNING: Could not connect to vector store. {e}")
    primary_vector_db = None
    secondary_vector_db = None


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
   - **NEW RULE - QUERY EXPANSION**: If the user asks for a region (e.g., "East", "North"), expand the 'keywords' to include actual neighborhood names (e.g., "Bedok, Tampines, Pasir Ris, Changi" for East). If they say "hawker", expand to "hawker centre, food court, local".
   - **NEW RULE - TARGETING**: If the user is looking for a SPECIFIC proper noun restaurant (e.g., "McDonalds", "Ah Seng Durian"), extract it into 'target_restaurant'. If they are asking for general cuisines or locations, leave it as null.
   - Return: {"intent": "SEARCH", "keywords": "expanded search terms", "target_restaurant": "Exact Name or null"}

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
        history_text = "None"
        if chat_history:
            recent_context = chat_history[-4:] 
            history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_context])

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
# 4. CORE FUNCTIONS (RAG & Generation)
# ==========================================
def search_cloud_db(query_text, user_profile=None, target_restaurant=None):
    if not primary_vector_db or not secondary_vector_db:
        print("   ❌ Databases not fully initialized.")
        return []
        
    filter_clauses = []

    # ONLY apply strict DB filters for dealbreakers (Health/Safety)
    if user_profile:
        if user_profile.get("diet") and len(user_profile["diet"]) > 0:
            filter_clauses.append({"diet": {"$eq": user_profile["diet"][0].lower()}})
        
        # TEMPORARILY DISABLED: Re-enable once 'allergens' metadata exists in DB
        # if user_profile.get("allergy") and len(user_profile["allergy"]) > 0:
        #     for allergen in user_profile["allergy"]:
        #         filter_clauses.append({"allergens": {"$ne": allergen.lower()}})

    search_filter = None
    if len(filter_clauses) == 1:
        search_filter = filter_clauses[0]
    elif len(filter_clauses) > 1:
        search_filter = {"$and": filter_clauses}

    try:
        # STAGE 1: Broad Vector Search
        if search_filter:
            print(f"   ⚙️ [Filter] Applying strict Dealbreaker rules: {search_filter}")
            primary_docs = primary_vector_db.similarity_search(query_text, k=40, filter=search_filter)
            secondary_docs = secondary_vector_db.similarity_search(query_text, k=40, filter=search_filter)
        else:
            primary_docs = primary_vector_db.similarity_search(query_text, k=40)
            secondary_docs = secondary_vector_db.similarity_search(query_text, k=40)
            
        # Combine all 80 results AND preserve their Vector Search ranking!
        all_docs = []
        for i, doc in enumerate(primary_docs):
            all_docs.append({"tier": "Primary Selection", "doc": doc, "base_score": 40 - i})
        for i, doc in enumerate(secondary_docs):
            all_docs.append({"tier": "Extended Selection", "doc": doc, "base_score": 40 - i})

        # STAGE 2: AI-Driven Lexical Reranker, Booster & Multiplier
        scored_results = []
        
        # Extract the user's favorite cuisines safely
        favorite_cuisines = []
        if user_profile and user_profile.get("cuisine"):
            favorite_cuisines = [c.lower() for c in user_profile["cuisine"]]
        
        # Define stop words so we don't boost generic filler
        stop_words = {"find", "me", "in", "the", "a", "some", "good", "food", "place", "restaurant", "for", "and", "or", "near", "around", "at"}
        
        # Clean commas out of the Router's expanded query
        clean_query = query_text.lower().replace(',', '').replace('.', '')
        query_words = set(clean_query.split())
        
        for item in all_docs:
            doc = item["doc"]
            restaurant_name = doc.metadata.get('name', 'Unknown Place').lower()
            category_tags = doc.metadata.get('category', '').lower()
            address_text = doc.metadata.get('address', '').lower()
            
            score = item["base_score"] 
            
            # --- 1. DYNAMIC EXACT NAME MATCHING ---
            if target_restaurant:
                target_lower = target_restaurant.lower()
                if target_lower in restaurant_name:
                    score += 500 # Explicit search always wins
                for word in target_lower.split():
                    if len(word) > 2 and word in restaurant_name: 
                        score += 100
            
            # --- 2. THE INTERSECTION MULTIPLIER ---
            category_hits = 0
            address_hits = 0
            
            for word in query_words:
                if len(word) > 2 and word not in stop_words:
                    if word in category_tags or word in restaurant_name:
                        category_hits += 1
                    if word in address_text:
                        address_hits += 1

            if category_hits > 0:
                score += 50
            if address_hits > 0:
                score += 50
                
            # THE JACKPOT: Perfect match for Cuisine AND Location
            if category_hits > 0 and address_hits > 0:
                score += 150 
            
            # --- 3. SOFT PREFERENCE BOOSTING ---
            for fav_cuisine in favorite_cuisines:
                if fav_cuisine in category_tags:
                    score += 20 
                    
            scored_results.append({
                "score": score,
                "name": doc.metadata.get('name', 'Unknown Place'),
                "category": doc.metadata.get('category', 'Food'),
                "tier": item["tier"],
                "description": doc.page_content
            })

        # Sort ALL results so the absolute best matches bubble to the top
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        # STAGE 3: Smart 2/2 Split (Balancing Primary and Secondary)
        primary_candidates = [r for r in scored_results if r["tier"] == "Primary Selection"]
        secondary_candidates = [r for r in scored_results if r["tier"] == "Extended Selection"]

        final_results = []
        
        # Grab the top 2 highest-scoring from each tier
        final_results.extend(primary_candidates[:2])
        final_results.extend(secondary_candidates[:2])

        # Fill the gaps if one database didn't have enough matches
        if len(final_results) < 4:
            remaining = [r for r in scored_results if r not in final_results]
            needed = 4 - len(final_results)
            final_results.extend(remaining[:needed])

        # Final sort to ensure the absolute highest scoring item is at the very top
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Clean up the output to send to LLM
        clean_results = [{"name": r["name"], "category": r["category"], "tier": r["tier"], "description": r["description"]} for r in final_results]
        
        print(f"   ✅ [Search] Reranked {len(all_docs)} docs. Top match score: {final_results[0]['score'] if final_results else 0}")
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
        
    # --- ANTI-HALLUCINATION LOCK ---
    else:
        messages.append({"role": "system", "content": "Database Results: NONE FOUND. You MUST apologize to the user and ask them to try different keywords. Do not recommend any real or fake places."})
    
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