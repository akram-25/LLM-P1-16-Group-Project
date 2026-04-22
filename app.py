import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, Response, stream_with_context
import chatbot_main as bot
import db

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# Initialise database schema on startup
try:
    db.init_db()
except Exception as e:
    print(f"Could not initialise database: {e}")


# AUTH ROUTES
@app.route("/login")
def login_page():
    # If already logged in, go to chat
    if session.get("user_id"):
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not username or not email or not password:
        return jsonify({"success": False, "error": "All fields are required"})

    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters"})

    user, error = db.register_user(username, email, password)

    if error:
        return jsonify({"success": False, "error": error})

    # Auto-login after registration
    session["user_id"] = user["user_id"]
    session["username"] = user["username"]
    session["new_user"] = True
    return jsonify({"success": True, "username": user["username"], "new_user": True})


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password are required"})

    user, error = db.login_user(username, password)

    if error:
        return jsonify({"success": False, "error": error})

    session["user_id"] = user["user_id"]
    session["username"] = user["username"]
    return jsonify({"success": True, "username": user["username"]})


@app.route("/auth/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


@app.route("/auth/me")
def auth_me():
    """Return current logged-in user info."""
    if session.get("user_id"):
        return jsonify({
            "logged_in": True,
            "username": session["username"],
            "user_id": session["user_id"]
        })
    return jsonify({"logged_in": False})


# GUEST MODE
@app.route("/chat/guest")
def guest_mode():
    session["user_id"] = "guest"
    session["username"] = "Guest"
    return redirect(url_for("home"))


# SETTINGS / PREFERENCES
@app.route("/settings")
def settings_page():
    if not session.get("user_id"):
        return redirect(url_for("login_page"))
    if session.get("user_id") == "guest":
        return redirect(url_for("home"))
    return render_template("settings.html")


@app.route("/settings/data")
def settings_data():
    """Return current user preferences as JSON."""
    user_id = session.get("user_id")
    if not user_id or user_id == "guest":
        return jsonify({"preferences": {}})
    prefs = db.get_preferences(user_id)
    return jsonify({"preferences": prefs})


@app.route("/settings/save", methods=["POST"])
def settings_save():
    """Save/replace all user preferences."""
    user_id = session.get("user_id")
    if not user_id or user_id == "guest":
        return jsonify({"success": False, "error": "Not logged in"})

    data = request.json
    prefs = data.get("preferences", {})

    db.save_preferences_bulk(user_id, prefs)
    db.set_onboarding_complete(user_id)

    # Clear the new_user flag
    session.pop("new_user", None)

    return jsonify({"success": True})


# MAIN CHAT ROUTES
@app.route("/")
def home():
    # Require login (or guest mode)
    if not session.get("user_id"):
        return redirect(url_for("login_page"))
    # Redirect new users to onboarding settings
    user_id = session.get("user_id")
    if user_id != "guest":
        if session.get("new_user") or not db.is_onboarding_complete(user_id):
            return redirect(url_for("settings_page") + "?onboarding=1")
    return render_template("index.html")


@app.route("/chat/history")
def get_history():
    """Load previous chat history for logged-in user."""
    user_id = session.get("user_id")
    if not user_id or user_id == "guest":
        return jsonify({"history": []})

    history = db.get_chat_history(user_id, limit=20)
    return jsonify({"history": history})


@app.route("/chat", methods=["POST"])
def chat():
    user_id = session.get("user_id")
    username = session.get("username", "Guest")
    is_guest = (user_id == "guest" or not user_id)

    user_input = request.json.get("message")

    if not user_input:
        return Response("Error: Empty message", mimetype='text/plain')

    try:
        # Load chat history from DB (empty for guests)
        if not is_guest:
            chat_history = db.get_chat_history(user_id, limit=10)
        else:
            chat_history = []

        # 1. Decide Intent
        decision = bot.get_intent(user_input, chat_history)
        intent = decision.get("intent")

        # 2. Load user preferences
        if not is_guest:
            user_profile = db.get_preferences(user_id)
            print(f"   [Prefs] Loaded for {username}: {user_profile}")
        else:
            user_profile = {}

        # 3. Streaming Generator Function
        def generate_stream():
            full_bot_reply = ""
            stream_source = []

            if intent == "SEARCH":
                original_query = decision.get("keywords", user_input)
                target_restaurant = decision.get("target_restaurant")

                if not is_guest:
                    db.save_search(user_id, original_query)

                # Reflection Loop (Max 2 Attempts)
                max_attempts = 2
                attempt = 1
                current_query = original_query
                best_db_results = []

                while attempt <= max_attempts:
                    print(f" [Attempt {attempt}] Searching for: {current_query}")
                    
                    # 1. Retrieve Data
                    db_results = bot.search_cloud_db(current_query, user_profile, target_restaurant)
                    best_db_results = db_results # Always save the latest attempt

                    # 2. Skip reflection if the user is looking for an exact restaurant name
                    if target_restaurant:
                        break

                    # 3. Observe and Reflect
                    evaluation = bot.evaluate_and_reflect(user_input, db_results)
                    print(f" [Reflection] Pass: {evaluation.get('pass')} | Reason: {evaluation.get('reasoning')}")

                    # 4. Decide
                    if evaluation.get("pass") == True:
                        break # If results are good, exit the loop 
                    else:
                        # 5. Revise and try again
                        current_query = evaluation.get("revised_query", current_query)
                        attempt += 1

                # Pass the best results to the Generator to stream to the user
                stream_source = bot.generate_response_with_history(
                    user_input, chat_history, context_data=best_db_results, user_profile=user_profile
                )

            elif intent == "SAVE_PREF":
                if not is_guest:
                    key = decision.get("key", "general")
                    value = decision.get("value", "true")
                    db.save_preference(user_id, key, value)
                    
                    user_profile_updated = db.get_preferences(user_id)
                    save_context = f"[SYSTEM NOTE: You just saved the user's {key} preference as '{value}'. Confirm this naturally in Singlish. Keep it short — 1-2 sentences.]"
                    chat_history_with_note = chat_history + [{"role": "system", "content": save_context}]
                    
                    stream_source = bot.generate_response_with_history(
                        user_input, chat_history_with_note, user_profile=user_profile_updated
                    )
                else:
                    # Fake a stream for static text
                    stream_source = ["Eh, I cannot save preferences for guest users lah. Create an account first!"]

            elif intent == "LIVE_SEARCH":
                query = decision.get("query", user_input)
                web_result = bot.search_live_web(query)
                live_context = [{"name": "Live Web Search", "description": web_result}]
                stream_source = bot.generate_response_with_history(
                    user_input, chat_history, context_data=live_context, user_profile=user_profile
                )

            elif intent == "SAVE_FAVORITE":
                restaurant_name = decision.get("restaurant_name", "").strip()
                if is_guest:
                    stream_source = ["Eh, I cannot save favourites for guest users lah. Create an account first!"]
                elif not restaurant_name:
                    stream_source = ["Hmm, I not sure which restaurant you want to save leh. Can you tell me the name again?"]
                else:
                    already_saved = not db.save_favorite(user_id, restaurant_name)
                    if already_saved:
                        note = f"[SYSTEM NOTE: The user asked to save '{restaurant_name}' but it is already in their favourites. Let them know in Singlish. Keep it short.]"
                    else:
                        note = f"[SYSTEM NOTE: You just saved '{restaurant_name}' to the user's favourites. Confirm this naturally in Singlish. Keep it short — 1-2 sentences.]"
                    chat_history_with_note = chat_history + [{"role": "system", "content": note}]
                    stream_source = bot.generate_response_with_history(
                        user_input, chat_history_with_note, user_profile=user_profile
                    )

            elif intent == "BLOCK":
                stream_source = ["Walao, I am a food chatbot leh. Ask me about food only. Don't ask me random things."]

            else:  # CHAT
                stream_source = bot.generate_response_with_history(
                    user_input, chat_history, user_profile=user_profile
                )

            # Yield the chunks to the frontend in real-time
            for chunk in stream_source:
                if chunk:
                    full_bot_reply += chunk
                    yield chunk

            # After the stream finishes, save the full conversation to DB (only if not BLOCK intent and not guest)
            if intent != "BLOCK" and not is_guest:
                db.save_chat_message(user_id, "user", user_input)
                db.save_chat_message(user_id, "assistant", full_bot_reply)

        # Return the generator wrapped in a Flask Response
        return Response(stream_with_context(generate_stream()), mimetype='text/plain')

    except Exception as e:
        print(f"Error in chat route: {e}")
        def error_stream():
            yield "Paiseh, my brain short circuit. Can try again?"
        return Response(stream_with_context(error_stream()), mimetype='text/plain')


@app.route("/chat/clear", methods=["POST"])
def clear_history():
    """Clear chat history for the current user."""
    user_id = session.get("user_id")
    if user_id and user_id != "guest":
        db.clear_chat_history(user_id)
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True)