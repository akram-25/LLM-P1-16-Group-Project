import os
import sys
import json
from functools import lru_cache
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

from prompts import INTENT_PROMPT, PERSONA_PROMPT, CRITIC_PROMPT

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
        return [{"error": "DB_OFFLINE"}] # <-- Send the distress signal
        
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

        # STAGE 2: AI-Driven Lexical Reranker, Quality Booster & Multiplier
        scored_results = []
        
        favorite_cuisines = []
        if user_profile and user_profile.get("cuisine"):
            favorite_cuisines = [c.lower() for c in user_profile["cuisine"]]
        
        stop_words = {"find", "me", "in", "the", "a", "some", "good", "food", "place", "restaurant", "for", "and", "or", "near", "around", "at", "best", "cheap"}
        
        clean_query = query_text.lower().replace(',', '').replace('.', '')
        query_words = set(clean_query.split())
        
        for item in all_docs:
            doc = item["doc"]
            restaurant_name = doc.metadata.get('name', 'Unknown Place').lower()
            category_tags = doc.metadata.get('category', '').lower()
            address_text = doc.metadata.get('address', '').lower()
            
            # Extract Rating and Review Count for Quality Control
            rating = float(doc.metadata.get('rating', 0.0))
            review_count = int(doc.metadata.get('user_rating_count', 0))
            
            score = item["base_score"] 
            
            # --- 1. DYNAMIC EXACT NAME MATCHING ---
            if target_restaurant:
                target_lower = target_restaurant.lower()
                if target_lower in restaurant_name:
                    score += 500 
                for word in target_lower.split():
                    if len(word) > 2 and word in restaurant_name: 
                        score += 100
            
            # --- 2. THE INTERSECTION MULTIPLIER ---
            category_hits = 0
            address_hits = 0
            
            for word in query_words:
                if len(word) > 2 and word not in stop_words:
                    # FIX: We NO LONGER check the restaurant name here! 
                    # We only check if the word is actually in the category/tags.
                    if word in category_tags:
                        category_hits += 1
                    if word in address_text:
                        address_hits += 1

            if category_hits > 0:
                score += 50
            if address_hits > 0:
                score += 50
                
            if category_hits > 0 and address_hits > 0:
                score += 150 
            
            # --- 3. SOFT PREFERENCE BOOSTING ---
            for fav_cuisine in favorite_cuisines:
                if fav_cuisine in category_tags:
                    score += 20 

            # --- 4. THE QUALITY MULTIPLIER (The "Best Options" Fix) ---
            # Reward places with genuinely good ratings
            if rating >= 4.0:
                # E.g., a 4.8 rating gets +16 points, a 4.1 gets +2 points
                score += (rating - 4.0) * 20 
                
                # Bonus points if they have a lot of reviews (proven consistency)
                if review_count > 100:
                    score += 10
                if review_count > 500:
                    score += 15
                    
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
        return [{"error": "DB_OFFLINE"}]
    

def evaluate_and_reflect(user_query, db_results):
    """
    The Reflection Agent: Grades the DB results against the user query.
    """
    # If the DB crashed or found absolutely nothing, it's an automatic fail
    if not db_results or db_results[0].get("error"):
        return {"pass": False, "revised_query": "broad popular food"}

    # Prepare the context for the Critic
    context_str = "Retrieved Restaurants:\n"
    for place in db_results:
        context_str += f"- {place['name']} ({place['category']}): {place['description'][:150]}...\n"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Use 3.5 for speed and low cost!
            messages=[
                {"role": "system", "content": CRITIC_PROMPT},
                {"role": "user", "content": f"USER QUERY: {user_query}\n\n{context_str}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"} 
        )
        evaluation = json.loads(response.choices[0].message.content)
        return evaluation
    except Exception as e:
        print(f"   [Critic Error] {e}")
        return {"pass": True} # Default to passing if the critic fails

def generate_response_with_history(new_user_input, chat_history, context_data=None, user_profile=None):
    messages = [{"role": "system", "content": PERSONA_PROMPT}]
    
    # Let the LLM see the user's saved preferences!
    if user_profile:
        messages.append({"role": "system", "content": f"User Preferences to keep in mind: {user_profile}"})
    
    if context_data:
        # --- NEW: Check for the Distress Signal ---
        if len(context_data) == 1 and context_data[0].get("error") == "DB_OFFLINE":
            messages.append({
                "role": "system", 
                "content": "CRITICAL ERROR: The restaurant database is completely offline. Apologize to the user (in your Singlish persona) and explain that your backend system is down right now, so you cannot search for any food."
            })
            
        # Check if it's Live Web Search data
        elif len(context_data) == 1 and context_data[0].get("name") == "Live Web Search":
            context_str = "Live Web Search Results (You may use this to answer the user):\n"
            context_str += f"- {context_data[0]['description']}\n"
            messages.append({"role": "system", "content": context_str})
            
        # Otherwise, it's a normal successful database search!
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
        temperature=0.7,
        stream=True
    )

    # Yield the text chunks as they arrive from OpenAI!
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# Initialize free web search tool
web_search_tool = DuckDuckGoSearchRun()

@lru_cache(maxsize=100)
def search_live_web(query):
    print(f"   🌐 [Agent] Searching the live web for: '{query}'...")
    try:
        # Ask DuckDuckGo for the top snippets
        return web_search_tool.run(query)
    except Exception as e:
        print(f"   ❌ Web Search Failed: {e}")
        return "Sorry, I couldn't find those details online right now."