import os
import json
import uuid
import threading
import psycopg2
import psycopg2.extras
import psycopg2.pool
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv('apikeys.env')

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST'),
    'database': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'port': int(os.getenv('POSTGRES_PORT', 5432))
}

_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(minconn=2, maxconn=10, **DB_CONFIG)
    return _pool


def get_connection():
    return _get_pool().getconn()


def release_connection(conn):
    _get_pool().putconn(conn)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id            VARCHAR(255) PRIMARY KEY,
            username           VARCHAR(255) UNIQUE NOT NULL,
            email              VARCHAR(255) UNIQUE NOT NULL,
            password_hash      VARCHAR(255),
            created_at         TIMESTAMP,
            last_login         TIMESTAMP,
            last_active        TIMESTAMP,
            onboarding_complete BOOLEAN DEFAULT FALSE
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            message_id SERIAL PRIMARY KEY,
            user_id    VARCHAR(255),
            session_id VARCHAR(255),
            role       VARCHAR(50),
            message    TEXT,
            timestamp  TIMESTAMP
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id               VARCHAR(255) PRIMARY KEY,
            price_preference      TEXT,
            preferred_location    TEXT,
            favorite_vibes        JSONB,
            favorite_cuisines     JSONB,
            dietary_restrictions  JSONB,
            allergens             JSONB,
            last_updated          TIMESTAMP
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            search_id     SERIAL PRIMARY KEY,
            user_id       VARCHAR(255),
            search_query  TEXT,
            search_type   VARCHAR(50),
            results_count INTEGER,
            searched_at   TIMESTAMP
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_favorites (
            favorite_id     SERIAL PRIMARY KEY,
            user_id         VARCHAR(255) NOT NULL,
            restaurant_name VARCHAR(255) NOT NULL,
            saved_at        TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE (user_id, restaurant_name)
        );
    """)

    # Add columns that may be missing on pre-existing databases
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255);")
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login TIMESTAMP;")
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarding_complete BOOLEAN DEFAULT FALSE;")

    conn.commit()
    cur.close()
    release_connection(conn)
    print("Database schema verified.")


# AUTHENTICATION
def register_user(username, email, password):
    # Register a new user
    # Returns (user_dict, None) on success or (None, error_message) on failure
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        # Check if username already exists
        cur.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            return None, "Username already taken"

        # Check if email already exists
        cur.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            return None, "Email already registered"

        user_id = str(uuid.uuid4())
        password_hash = generate_password_hash(password)
        now = datetime.now()

        cur.execute("""
            INSERT INTO users (user_id, username, email, password_hash, created_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING user_id, username, email, created_at
        """, (user_id, username, email, password_hash, now))

        user_row = cur.fetchone()
        conn.commit()

        return dict(user_row), None

    except Exception as e:
        conn.rollback()
        print(f"[DB Register Error] {e}")
        return None, f"Registration failed: {str(e)}"
    finally:
        cur.close()
        release_connection(conn)


def login_user(username, password):
    # Authenticate a user
    # Returns (user_dict, None) on success or (None, error_message) on failure
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()

        if not user:
            return None, "Invalid username or password"

        if not user.get('password_hash'):
            return None, "Account not set up for login. Please register."

        if not check_password_hash(user['password_hash'], password):
            return None, "Invalid username or password"

        # Update last_login and last_active
        now = datetime.now()
        cur.execute(
            "UPDATE users SET last_login = %s, last_active = %s WHERE user_id = %s",
            (now, now, user['user_id'])
        )
        conn.commit()

        return {
            'user_id': user['user_id'],
            'username': user['username'],
            'email': user['email']
        }, None

    except Exception as e:
        print(f"[DB Login Error] {e}")
        return None, f"Login failed: {str(e)}"
    finally:
        cur.close()
        release_connection(conn)


# CHAT HISTORY
# Actual table schema:
#   message_id SERIAL PK, user_id, session_id, role, message, timestamp, tokens_used
def save_chat_message(user_id, role, content, session_id=None):
    # Save a single chat message to the chat_history table
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_history (user_id, session_id, role, message, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """, (user_id, session_id, role, content, datetime.now()))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[DB Chat Save Error] {e}")
    finally:
        cur.close()
        release_connection(conn)


def get_chat_history(user_id, limit=10):
    # Retrieve last N chat messages for a user
    # Returns list of {"role": ..., "content": ...} in chronological order
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("""
            SELECT role, message FROM chat_history
            WHERE user_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """, (user_id, limit))
        rows = cur.fetchall()
        # Reverse to get chronological order
        return [{"role": r['role'], "content": r['message']} for r in reversed(rows)]
    except Exception as e:
        print(f"[DB Chat Load Error] {e}")
        return []
    finally:
        cur.close()
        release_connection(conn)


def clear_chat_history(user_id):
    # Clear all chat history for a user
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM chat_history WHERE user_id = %s", (user_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[DB Chat Clear Error] {e}")
    finally:
        cur.close()
        release_connection(conn)



# USER PREFERENCES (PostgreSQL-backed)
# Actual table schema (single row per user with JSONB columns):
#   user_id PK, student_status, price_preference, budget_max,
#   favorite_vibes (jsonb), favorite_cuisines (jsonb),
#   dietary_restrictions (jsonb), allergens (jsonb),
#   preferred_location, last_updated
def save_preference(user_id, pref_key, pref_value):
    # Map chatbot preference keys to actual DB JSONB columns
    KEY_MAP = {
        "diet": "dietary_restrictions",
        "allergy": "allergens",
        "cuisine": "favorite_cuisines",
        "location": "preferred_location",
        "budget": "price_preference",
        "spice": "favorite_vibes",
    }

    db_column = KEY_MAP.get(pref_key)
    if not db_column:
        print(f"[Pref] Unknown preference key: {pref_key}, skipping.")
        return

    conn = get_connection()
    cur = conn.cursor()
    try:
        # Ensure a row exists for this user
        cur.execute("SELECT user_id FROM user_preferences WHERE user_id = %s", (user_id,))
        if not cur.fetchone():
            cur.execute("INSERT INTO user_preferences (user_id, last_updated) VALUES (%s, %s)", (user_id, datetime.now()))

        # For JSONB columns, append to the array
        if db_column in ("dietary_restrictions", "allergens", "favorite_cuisines", "favorite_vibes"):
            cur.execute(f"""
                UPDATE user_preferences
                SET {db_column} = COALESCE({db_column}, '[]'::jsonb) || %s::jsonb,
                    last_updated = %s
                WHERE user_id = %s
            """, (json.dumps([pref_value]), datetime.now(), user_id))
        else:
            # For simple text columns (preferred_location, price_preference)
            cur.execute(f"""
                UPDATE user_preferences
                SET {db_column} = %s, last_updated = %s
                WHERE user_id = %s
            """, (pref_value, datetime.now(), user_id))

        conn.commit()
        print(f"[Memory] Saved {pref_key} → {db_column}: {pref_value} for user {user_id}")
    except Exception as e:
        conn.rollback()
        print(f"[DB Pref Save Error] {e}")
    finally:
        cur.close()
        release_connection(conn)


def get_preferences(user_id):
    # Load all preferences for a user
    # Returns dict like: {"diet": ["halal"], "allergy": ["peanuts"], "cuisine": ["Japanese"]}
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("SELECT * FROM user_preferences WHERE user_id = %s", (user_id,))
        row = cur.fetchone()

        if not row:
            return {}

        prefs = {}
        # Map DB columns back to friendly keys
        if row.get('dietary_restrictions'):
            prefs['diet'] = row['dietary_restrictions'] if isinstance(row['dietary_restrictions'], list) else []
        if row.get('allergens'):
            prefs['allergy'] = row['allergens'] if isinstance(row['allergens'], list) else []
        if row.get('favorite_cuisines'):
            prefs['cuisine'] = row['favorite_cuisines'] if isinstance(row['favorite_cuisines'], list) else []
        if row.get('favorite_vibes'):
            prefs['spice'] = row['favorite_vibes'] if isinstance(row['favorite_vibes'], list) else []
        if row.get('preferred_location'):
            prefs['location'] = [row['preferred_location']]
        if row.get('price_preference'):
            prefs['budget'] = [row['price_preference']]
        if row.get('budget_max'):
            prefs['budget_max'] = row['budget_max']

        return prefs
    except Exception as e:
        print(f"[DB Pref Load Error] {e}")
        return {}
    finally:
        cur.close()
        release_connection(conn)


def save_preferences_bulk(user_id, prefs_dict):
    # Replace all preferences for a user from the settings page
    # prefs_dict format from frontend:
    #  {"allergy": ["nuts", "seafood"], "cuisine": ["Japanese", "Malay"],
    #   "diet": ["Halal"], "budget": ["$10-$20"], "spice": ["Mild"],
    #   "location": ["Ang Mo Kio"], "notes": ["I love desserts"]}
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Ensure a row exists
        cur.execute("SELECT user_id FROM user_preferences WHERE user_id = %s", (user_id,))
        if not cur.fetchone():
            cur.execute("INSERT INTO user_preferences (user_id) VALUES (%s)", (user_id,))

        # Map frontend keys to actual DB columns
        allergens = json.dumps(prefs_dict.get('allergy', []))
        cuisines = json.dumps(prefs_dict.get('cuisine', []))
        diet = json.dumps(prefs_dict.get('diet', []))
        vibes = json.dumps(prefs_dict.get('spice', []))
        location = prefs_dict.get('location', [None])[0] if prefs_dict.get('location') else None
        budget = prefs_dict.get('budget', [None])[0] if prefs_dict.get('budget') else None
        notes = prefs_dict.get('notes', [None])[0] if prefs_dict.get('notes') else None

        cur.execute("""
            UPDATE user_preferences
            SET allergens = %s::jsonb,
                favorite_cuisines = %s::jsonb,
                dietary_restrictions = %s::jsonb,
                favorite_vibes = %s::jsonb,
                preferred_location = %s,
                price_preference = %s,
                last_updated = %s
            WHERE user_id = %s
        """, (allergens, cuisines, diet, vibes, location, budget, datetime.now(), user_id))

        conn.commit()
        print(f"[Memory] Saved all preferences for user {user_id}")
    except Exception as e:
        conn.rollback()
        print(f"[DB Bulk Pref Save Error] {e}")
    finally:
        cur.close()
        release_connection(conn)


def is_onboarding_complete(user_id):
    # Check if the user has completed onboarding
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT onboarding_complete FROM users WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
        return row[0] if row else False
    except Exception as e:
        print(f"[DB Onboarding Check Error] {e}")
        return False
    finally:
        cur.close()
        release_connection(conn)


def set_onboarding_complete(user_id):
    # Mark the user's onboarding as complete
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE users SET onboarding_complete = TRUE WHERE user_id = %s", (user_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[DB Onboarding Set Error] {e}")
    finally:
        cur.close()
        release_connection(conn)


# SEARCH HISTORY
# Actual table schema:
#   search_id SERIAL PK, user_id, search_query, search_type, results_count, searched_at

def save_search(user_id, query, search_type="chatbot", results_count=0):
    # Log a search query.
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO search_history (user_id, search_query, search_type, results_count, searched_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (user_id, query, search_type, results_count, datetime.now()))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[DB Search Save Error] {e}")
    finally:
        cur.close()
        release_connection(conn)


# USER FAVORITES
def save_favorite(user_id, restaurant_name):
    """
    Save a restaurant to the user's favourites.
    Silently ignores duplicates (UNIQUE constraint).
    Returns True if newly saved, False if already existed.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO user_favorites (user_id, restaurant_name, saved_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, restaurant_name) DO NOTHING
        """, (user_id, restaurant_name, datetime.now()))
        saved = cur.rowcount > 0
        conn.commit()
        print(f"   [Favourites] {'Saved' if saved else 'Already saved'}: {restaurant_name} for user {user_id}")
        return saved
    except Exception as e:
        conn.rollback()
        print(f"[DB Favourites Save Error] {e}")
        return False
    finally:
        cur.close()
        release_connection(conn)


def get_favorites(user_id):
    """Return a list of saved restaurant names for a user, newest first."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("""
            SELECT restaurant_name, saved_at FROM user_favorites
            WHERE user_id = %s
            ORDER BY saved_at DESC
        """, (user_id,))
        return [{"restaurant_name": r["restaurant_name"], "saved_at": str(r["saved_at"])} for r in cur.fetchall()]
    except Exception as e:
        print(f"[DB Favourites Load Error] {e}")
        return []
    finally:
        cur.close()
        release_connection(conn)