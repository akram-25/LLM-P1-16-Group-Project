from flask import Flask, render_template, request, jsonify
import chatbot_main as bot

app = Flask(__name__)

# Global chat history
chat_history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    user_input = request.json.get("message")
    
    if not user_input:
        return jsonify({"reply": "Error: Empty message"})

    try:
        # 1. Decide Intent
        decision = bot.get_intent(user_input)
        intent = decision.get("intent")
        
        db_results = None
        bot_reply = ""

        # 2. Execute Logic
        if intent == "SEARCH":
            search_query = decision.get("keywords", user_input)
            
            # --- USING THE SELF-RAG REFLECTION SEARCH ---
            db_results = bot.reflective_search(user_input, search_query)
            
            bot_reply = bot.generate_response_with_history(user_input, chat_history, db_results)
            
        elif intent == "SAVE_PREF":
            key = decision.get("key", "general")
            value = decision.get("value", "true")
            bot.save_user_preference("Akram", key, value)
            
            if key == "name":
                bot_reply = f"Nice to meet you, {value}! I will remember that."
            else:
                bot_reply = f"Ok can. I saved that your {key} is {value}."
                
        elif intent == "BLOCK":
            bot_reply = "Walao, I am a food chatbot leh. Ask me about food only. Don't ask me random things."
            
        else: # CHAT
            bot_reply = bot.generate_response_with_history(user_input, chat_history)

        # 3. Update History
        if intent != "BLOCK":
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": bot_reply})
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        return jsonify({"reply": bot_reply})

    except Exception as e:
        print(f"Error in chat route: {e}")
        return jsonify({"reply": "Paiseh, my brain short circuit. Can try again?"})

if __name__ == "__main__":
    app.run(debug=True)