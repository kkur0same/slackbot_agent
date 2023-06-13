import os
import yaml
#import asyncio
import logging
from flask import Flask
from concurrent.futures import ThreadPoolExecutor
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from slack_sdk.web import WebClient
from slack_bolt.adapter.flask import SlackRequestHandler
from lazy_model import LazyModel, LazyEmbedding


def setup_app():
    # Load environment variables from .env file
    load_dotenv(find_dotenv())

    # Set Slack API credentials
    slack_bot_token = os.environ["SLACK_BOT_TOKEN"]
    slack_signing_secret = os.environ["SLACK_SIGNING_SECRET"]
    slack_user_token = os.environ["SLACK_USER_TOKEN"]

    # Initialize the Slack app
    app = App(token=slack_bot_token)

    # Initialize the Flask app
    flask_app = Flask(__name__)
    handler = SlackRequestHandler(app)

    # Initialize the Slack client
    slack_client = WebClient(token=slack_bot_token)
    bot_id = slack_client.api_call("auth.test")["user_id"]
    slack_user_client = WebClient(token=slack_user_token)

    # Initialize executor
    executor = ThreadPoolExecutor()
    
    logging.basicConfig(
    level=logging.INFO,  # This will log all levels from DEBUG and above.
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('slack_bot.log'),  # File handler - logs to a file.
        logging.StreamHandler()  # Stream handler - logs to console.
        ]
    )

    return app, flask_app, handler, slack_client, bot_id, slack_user_token, slack_user_client, executor



def load_config(file_name="config.yaml"):
    with open(file_name, "r") as file:
        config = yaml.safe_load(file)

    models_config = config["models"]
    embeddings_config = config["embeddings"]
    models = {}
    for model_config in models_config:
        model_name = model_config['name']
        models[model_name] = LazyModel(model_config)
    embeddings = {}
    for embedding_config in embeddings_config:
        embedding_name = embedding_config['name']
        embeddings[embedding_name] = LazyEmbedding(embedding_config)

    available_models = [model["name"] for model in config["models"]]
    available_embeddings = [embedding["name"] for embedding in config["embeddings"]]

    return config, models, embeddings, available_models, available_embeddings