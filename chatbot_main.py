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

# 1. SETUP & CONNECTION
load_dotenv('apikeys.env')

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

PG_CONNECTION = (
    f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

print("Connecting to PostgreSQL vector store...")

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
    print("Vector store connected.")
except Exception as e:
    print(f"\nWARNING: Could not connect to vector store. {e}")
    primary_vector_db = None
    secondary_vector_db = None


# 2. USER MEMORY (Long-Term Storage)
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

# 4. CORE FUNCTIONS (RAG & Generation)
def search_cloud_db(query_text, user_profile=None, target_restaurant=None):
    if not primary_vector_db or not secondary_vector_db:
        print(" Databases not fully initialised.")
        return [{"error": "DB_OFFLINE"}]

    try:
        # Stage 1: Broad Vector Search (use original query + diet is enforced in reranker)
        primary_docs = primary_vector_db.similarity_search(query_text, k=40)
        secondary_docs = secondary_vector_db.similarity_search(query_text, k=40)
            
        # Combine all 80 results and preserve their Vector Search ranking
        all_docs = []
        for i, doc in enumerate(primary_docs):
            all_docs.append({"tier": "Primary Selection", "doc": doc, "base_score": 40 - i})
        for i, doc in enumerate(secondary_docs):
            all_docs.append({"tier": "Extended Selection", "doc": doc, "base_score": 40 - i})

        # Stage 2: AI-Driven Lexical Reranker, Quality Booster and Multiplier
        scored_results = []
        
        favorite_cuisines = []
        if user_profile and user_profile.get("cuisine"):
            favorite_cuisines = [c.lower() for c in user_profile["cuisine"]]
        
        stop_words = {
            "find", "me", "in", "the", "a", "some", "good", "food", "place", "restaurant",
            "for", "and", "or", "near", "around", "at", "best", "cheap", "recommend",
            "want", "area", "looking", "craving", "try", "suggest", "show", "give",
            "i", "can", "you", "please", "like", "any", "something", "spots", "spot",
            "options", "option", "get", "know", "tell", "about", "with",
        }

        diet_terms = [d.lower() for d in (user_profile.get("diet", []) if user_profile else [])]

        # Coordinate-based region bounds for Singapore
        # Used to enforce location when the user specifies a region
        REGION_BOUNDS = {
            "west":      {"lat": (1.28, 1.38), "lon": (103.60, 103.82)},
            "east":      {"lat": (1.29, 1.41), "lon": (103.86, 104.03)},
            "north":     {"lat": (1.39, 1.47), "lon": (103.74, 103.88)},
            "central":   {"lat": (1.27, 1.37), "lon": (103.81, 103.88)},
            "northeast": {"lat": (1.34, 1.43), "lon": (103.87, 103.98)},
        }

        clean_query = query_text.lower().replace(',', '').replace('.', '')
        query_words = set(clean_query.split())

        # Detect if the user specified a region
        detected_region = None
        for region in REGION_BOUNDS:
            if region in query_words:
                detected_region = region
                break
        
        for item in all_docs:
            doc = item["doc"]
            restaurant_name = doc.metadata.get('name', 'Unknown Place').lower()
            category_tags = doc.metadata.get('category', '').lower()
            address_text = doc.metadata.get('address', '').lower()
            
            # Extract Rating and Review Count for Quality Control
            rating = float(doc.metadata.get('rating', 0.0))
            review_count = int(doc.metadata.get('user_rating_count', 0))
            
            score = item["base_score"] 
            
            # 1. DYNAMIC EXACT NAME MATCHING
            if target_restaurant:
                target_lower = target_restaurant.lower()
                if target_lower in restaurant_name:
                    score += 500 
                for word in target_lower.split():
                    if len(word) > 2 and word in restaurant_name: 
                        score += 100
            
            # 2. THE INTERSECTION MULTIPLIER
            category_hits = 0
            address_hits = 0
            
            for word in query_words:
                if len(word) > 2 and word not in stop_words:
                    # Only check if the word is actually in the category/tags
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
            
            # 3. DIETARY SCORING
            # Boost restaurants whose category contains the user's diet term (e.g. "halal","vegetarian")
            # Penalise those that don't
            # Penalty is intentionally larger than a single location bonus so diet compliance beats bare location match
            for diet in diet_terms:
                if diet in category_tags:
                    score += 80
                else:
                    score -= 100

            # 4. COORDINATE-BASED REGION ENFORCEMENT
            # If the user asked for a specific region (e.g. "west"), use the restaurant's lat/lon to confirm it is actually in that region
            # A wrong-region restaurant gets a heavy penalty that overrides diet and quality bonuses
            if detected_region:
                bounds = REGION_BOUNDS[detected_region]
                try:
                    lat = float(doc.metadata.get('latitude', 0))
                    lon = float(doc.metadata.get('longitude', 0))
                    lat_ok = bounds["lat"][0] <= lat <= bounds["lat"][1]
                    lon_ok = bounds["lon"][0] <= lon <= bounds["lon"][1]
                    if lat_ok and lon_ok:
                        score += 200
                    else:
                        score -= 200
                except (TypeError, ValueError):
                    pass

            # 5. SOFT PREFERENCE BOOSTING
            for fav_cuisine in favorite_cuisines:
                if fav_cuisine in category_tags:
                    score += 20

            # 6. THE QUALITY MULTIPLIER
            # Reward places with genuinely good ratings
            if rating >= 4.0:
                # E.g., a 4.8 rating gets +16 points, a 4.1 gets +2 points
                score += (rating - 4.0) * 20 
                
                # Bonus points if they have a lot of reviews
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

        # Sort ALL results so the absolute best matches gets pushed to the top
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        # STAGE 3: Smart 2/2 Split (Balancing Primary and Secondary)
        primary_candidates = [r for r in scored_results if r["tier"] == "Primary Selection"]
        secondary_candidates = [r for r in scored_results if r["tier"] == "Extended Selection"]

        final_results = []
        
        # Grab the top 2 highest-scoring from each tier
        final_results.extend(primary_candidates[:2])
        final_results.extend(secondary_candidates[:2])

        # Fill the gaps if one database doesn't have enough matches
        if len(final_results) < 4:
            remaining = [r for r in scored_results if r not in final_results]
            needed = 4 - len(final_results)
            final_results.extend(remaining[:needed])

        # Final sort to ensure the absolute highest scoring item is at the very top
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Clean up the output to send to LLM
        clean_results = [{"name": r["name"], "category": r["category"], "tier": r["tier"], "description": r["description"]} for r in final_results]
        
        print(f"[Search] Reranked {len(all_docs)} docs. Top match score: {final_results[0]['score'] if final_results else 0}")
        return clean_results
        
    except Exception as e:
        print(f"DB Search Failed: {e}")
        return [{"error": "DB_OFFLINE"}]
    

def evaluate_and_reflect(user_query, db_results):
    # The Reflection Agent: Grades the DB results against the user query.
    # If the DB crashed or found absolutely nothing, it has failed
    if not db_results or db_results[0].get("error"):
        return {"pass": False, "revised_query": "broad popular food"}

    # Prepare the context for the Critic
    context_str = "Retrieved Restaurants:\n"
    for place in db_results:
        context_str += f"- {place['name']} ({place['category']}): {place['description'][:150]}...\n"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
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

    # Inject preferences and hard dietary enforcement rule
    if user_profile:
        messages.append({"role": "system", "content": f"User Preferences to keep in mind: {user_profile}"})
        diet_terms = user_profile.get("diet", [])
        if diet_terms:
            diet_str = ", ".join(diet_terms)
            messages.append({"role": "system", "content": (
                f"CRITICAL DIETARY RULE: This user requires {diet_str} food. "
                f"A restaurant is ONLY acceptable if it is PRIMARILY {diet_str} — "
                f"meaning its core identity is {diet_str} (e.g. a vegetarian restaurant, a halal-certified eatery). "
                f"A meat-focused or non-certified restaurant that merely offers a couple of {diet_str} options is NOT acceptable. "
                f"You MUST silently skip any restaurant that does not clearly meet this standard. "
                f"If none of the retrieved results are suitable, say so honestly and ask the user to try different keywords."
            )})
    
    if context_data:
        # Check for the Distress Signal
        if len(context_data) == 1 and context_data[0].get("error") == "DB_OFFLINE":
            messages.append({
                "role": "system", 
                "content": "CRITICAL ERROR: The restaurant database is completely offline. Apologise to the user (in your Singlish persona) and explain that your backend system is down right now, so you cannot search for any food."
            })
            
        # Check if it's Live Web Search data
        elif len(context_data) == 1 and context_data[0].get("name") == "Live Web Search":
            context_str = "Live Web Search Results (You may use this to answer the user):\n"
            context_str += f"- {context_data[0]['description']}\n"
            messages.append({"role": "system", "content": context_str})
            
        # Else it's a normal successful database search
        else:
            context_str = "Database Results:\n"
            for place in context_data:
                context_str += f"- {place['name']} ({place['category']} - {place['tier']}): {place['description']}\n"
            messages.append({"role": "system", "content": context_str})
        
    # ANTI-HALLUCINATION LOCK
    else:
        messages.append({"role": "system", "content": "Database Results: NONE FOUND. You MUST apologise to the user and ask them to try different keywords. Do not recommend any real or fake places."})
    
    messages.extend(chat_history)
    messages.append({"role": "user", "content": new_user_input})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    # Yield the text chunks as they arrive from OpenAI
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# Initialise web search tool
web_search_tool = DuckDuckGoSearchRun()

@lru_cache(maxsize=100)
def search_live_web(query):
    print(f"[Agent] Searching the live web for: '{query}'...")
    try:
        return web_search_tool.run(query)
    except Exception as e:
        print(f"Web Search Failed: {e}")
        return "Sorry, I couldn't find those details online right now."