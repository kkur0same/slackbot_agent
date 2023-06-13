from slack_db import setup_database
from utils import load_config, setup_app
from flask import Flask, request
from setup_app import setup_app
from events_handler import *
from transformers import AutoTokenizer

setup_database()
config, models, embeddings, available_models, available_embeddings = load_config()
app, flask_app, handler, slack_client, bot_id, slack_user_token, slack_user_client, executor = setup_app()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Attach event handlers
app.event("message")(handle_message_events)
app.event("app_mention")(handle_mentions)
app.event("reaction_added")(handle_reaction)
app.command("/safety_mode")(activate_safety_mode)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """
    return handler.handle(request)


periodic_cleanup()


if __name__ == "__main__":
    flask_app.run(debug=True, port=5002)

