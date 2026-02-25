import os
import uuid
import psycopg2
import psycopg2.extras
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


def get_connection():
    """Get a new PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    """
    Ensure all required tables and columns exist.
    Safe to run multiple times (uses IF NOT EXISTS).
    """
    conn = get_connection()
    cur = conn.cursor()

    # Add password_hash, last_login, and onboarding_complete columns to users table if missing
    cur.execute("""
        ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255);
    """)
    cur.execute("""
        ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login TIMESTAMP;
    """)
    cur.execute("""
        ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarding_complete BOOLEAN DEFAULT FALSE;
    """)

    # Ensure chat_history table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(100) REFERENCES users(user_id),
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Ensure user_preferences table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(100) REFERENCES users(user_id),
            preference_key VARCHAR(100) NOT NULL,
            preference_value VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Ensure search_history table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(100) REFERENCES users(user_id),
            query TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("   ✅ Database schema verified.")


# ==========================================
# AUTHENTICATION
# ==========================================
def register_user(username, email, password):
    """
    Register a new user.
    Returns (user_dict, None) on success, or (None, error_message) on failure.
    """
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
        print(f"   ❌ [DB Register Error] {e}")
        return None, f"Registration failed: {str(e)}"
    finally:
        cur.close()
        conn.close()


def login_user(username, password):
    """
    Authenticate a user.
    Returns (user_dict, None) on success, or (None, error_message) on failure.
    """
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

        # Update last_login
        cur.execute(
            "UPDATE users SET last_login = %s WHERE user_id = %s",
            (datetime.now(), user['user_id'])
        )
        conn.commit()

        return {
            'user_id': user['user_id'],
            'username': user['username'],
            'email': user['email']
        }, None

    except Exception as e:
        print(f"   ❌ [DB Login Error] {e}")
        return None, f"Login failed: {str(e)}"
    finally:
        cur.close()
        conn.close()


# ==========================================
# CHAT HISTORY
# ==========================================
def save_chat_message(user_id, role, content):
    """Save a single chat message to the database."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_history (user_id, role, content, created_at)
            VALUES (%s, %s, %s, %s)
        """, (user_id, role, content, datetime.now()))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"   ❌ [DB Chat Save Error] {e}")
    finally:
        cur.close()
        conn.close()


def get_chat_history(user_id, limit=10):
    """
    Retrieve last N chat messages for a user.
    Returns list of {"role": ..., "content": ...} in chronological order.
    """
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("""
            SELECT role, content FROM chat_history
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (user_id, limit))
        rows = cur.fetchall()
        # Reverse to get chronological order
        return [{"role": r['role'], "content": r['content']} for r in reversed(rows)]
    except Exception as e:
        print(f"   ❌ [DB Chat Load Error] {e}")
        return []
    finally:
        cur.close()
        conn.close()


def clear_chat_history(user_id):
    """Clear all chat history for a user."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM chat_history WHERE user_id = %s", (user_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"   ❌ [DB Chat Clear Error] {e}")
    finally:
        cur.close()
        conn.close()


# ==========================================
# USER PREFERENCES (PostgreSQL-backed)
# ==========================================
def save_preference(user_id, pref_key, pref_value):
    """Save a user preference. Avoids duplicates."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Check for duplicate
        cur.execute("""
            SELECT id FROM user_preferences
            WHERE user_id = %s AND preference_key = %s AND LOWER(preference_value) = LOWER(%s)
        """, (user_id, pref_key, pref_value))

        if not cur.fetchone():
            cur.execute("""
                INSERT INTO user_preferences (user_id, preference_key, preference_value, created_at)
                VALUES (%s, %s, %s, %s)
            """, (user_id, pref_key, pref_value, datetime.now()))
            conn.commit()
            print(f"   [Memory] Saved {pref_key}: {pref_value} for user {user_id}")
        else:
            print(f"   [Memory] Preference already exists, skipping.")
    except Exception as e:
        conn.rollback()
        print(f"   ❌ [DB Pref Save Error] {e}")
    finally:
        cur.close()
        conn.close()


def get_preferences(user_id):
    """
    Load all preferences for a user.
    Returns dict like: {"diet": ["halal", "no pork"], "allergy": ["seafood"]}
    """
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("""
            SELECT preference_key, preference_value FROM user_preferences
            WHERE user_id = %s
            ORDER BY created_at ASC
        """, (user_id,))
        rows = cur.fetchall()

        prefs = {}
        for row in rows:
            key = row['preference_key']
            value = row['preference_value']
            if key not in prefs:
                prefs[key] = []
            prefs[key].append(value)
        return prefs
    except Exception as e:
        print(f"   ❌ [DB Pref Load Error] {e}")
        return {}
    finally:
        cur.close()
        conn.close()


def delete_preferences_by_key(user_id, pref_key):
    """Delete all preferences for a given key."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            DELETE FROM user_preferences
            WHERE user_id = %s AND preference_key = %s
        """, (user_id, pref_key))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"   ❌ [DB Pref Delete Error] {e}")
    finally:
        cur.close()
        conn.close()


def save_preferences_bulk(user_id, prefs_dict):
    """
    Replace all preferences for a user with the given dict.
    prefs_dict format: {"allergy": ["nuts", "seafood"], "cuisine": ["Japanese", "Malay"], ...}
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Delete all existing preferences for this user
        cur.execute("DELETE FROM user_preferences WHERE user_id = %s", (user_id,))

        # Insert new ones
        now = datetime.now()
        for key, values in prefs_dict.items():
            if isinstance(values, str):
                values = [values]
            for val in values:
                val = val.strip()
                if val:  # Skip empty strings
                    cur.execute("""
                        INSERT INTO user_preferences (user_id, preference_key, preference_value, created_at)
                        VALUES (%s, %s, %s, %s)
                    """, (user_id, key, val, now))

        conn.commit()
        print(f"   [Memory] Saved all preferences for user {user_id}")
    except Exception as e:
        conn.rollback()
        print(f"   ❌ [DB Bulk Pref Save Error] {e}")
    finally:
        cur.close()
        conn.close()


def is_onboarding_complete(user_id):
    """Check if the user has completed onboarding."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT onboarding_complete FROM users WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
        return row[0] if row else False
    except Exception as e:
        print(f"   ❌ [DB Onboarding Check Error] {e}")
        return False
    finally:
        cur.close()
        conn.close()


def set_onboarding_complete(user_id):
    """Mark the user's onboarding as complete."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE users SET onboarding_complete = TRUE WHERE user_id = %s", (user_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"   ❌ [DB Onboarding Set Error] {e}")
    finally:
        cur.close()
        conn.close()


# ==========================================
# SEARCH HISTORY
# ==========================================
def save_search(user_id, query):
    """Log a search query."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO search_history (user_id, query, created_at)
            VALUES (%s, %s, %s)
        """, (user_id, query, datetime.now()))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"   ❌ [DB Search Save Error] {e}")
    finally:
        cur.close()
        conn.close()
