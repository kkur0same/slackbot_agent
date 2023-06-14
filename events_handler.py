from threading import Lock
from utils import *
import json
from setup_app import app, bot_id, slack_client
from chatbot import initialize_chatbot
from slack_db import record_interaction_in_database
import time
from threading import Timer


user_chat_states_lock = Lock()
user_chat_states = {}

class ChatState:
    def __init__(self):
        self.safety_mode = False
        self.thread_histories = {}
        self.last_active = time.time()
        self.chatbot_instance = None 

    def set_safety_mode(self, mode):
        self.safety_mode = mode

    def get_safety_mode(self):
        return self.safety_mode
    
    def get_thread_history(self, thread_id):
        return self.thread_histories.get(thread_id, [])
    
    def set_thread_history(self, thread_id, history):
        self.thread_histories[thread_id] = history

    def set_chatbot_instance(self, chatbot):
        self.chatbot_instance = chatbot
 
    def get_chatbot_instance(self):
        return self.chatbot_instance
    
    def update_last_active(self):
        self.last_active = time.time()


def handle_message_events(body, logger):
    logger.info(body)

def handle_mentions(body, say, logger, config):
    user_message = body["event"]["text"]
    bot_user_id = bot_id
    channel_id = body["event"]["channel"]
    mention = f"<@{bot_user_id}>"
    user_message = user_message.replace(mention, "").strip()
    user_id = body["event"]["user"]
    ts = body["event"]["ts"]
    thread_ts = body["event"]["thread_ts"] if "thread_ts" in body["event"] else ts
    is_new_conversation = (ts == thread_ts)

    with user_chat_states_lock:
        if user_id not in user_chat_states:
            user_chat_states[user_id] = ChatState()      

    chat_state = user_chat_states[user_id]
    chat_state.update_last_active()
    safety_mode = chat_state.get_safety_mode()
    history = chat_state.get_thread_history(thread_ts)
    session_type = 'mention'

    if is_new_conversation or chat_state.get_chatbot_instance() is None:
        model_name, embedding_name = get_model_and_embedding(safety_mode, user_message, say, ts)
        lazy_model, lazy_embedding = get_lazy_model_and_embedding(model_name, embedding_name)
        qa = initialize_chatbot(lazy_model, lazy_embedding, config)
        chat_state.set_chatbot_instance(qa)
    else:
        qa = chat_state.get_chatbot_instance()

    user_message = clean_user_message(user_message)
    response, answer, history = get_response_and_history(qa, user_message, history)
    chat_state.set_thread_history(thread_ts, history)
    formatted_response, formatted_sources = format_response(response, answer)

    if safety_mode:
        print(safety_mode)
        channels = ["<intentially left blank>"]
        context = search_messages(user_message, channels, user_id)
        print(context)
        for match in context:
            ref_channel_name = match["channel_name"]
            ref_channel_id = match["channel_id"]
            permalink = match["permalink"] 
        # Add a message with the channel, thread and matching message
            formatted_response += f"\n\nYou might also find more information in the thread <{permalink}|link> in Slack channel <#{ref_channel_id}|{ref_channel_name}>."

    bot_response = slack_client.chat_postMessage(channel=channel_id, text=formatted_response, thread_ts=ts)
    bot_response_ts = bot_response['ts']
    history_string = json.dumps(history)
    
    logger.info(f"User ID: {body['event']['user']}")
    logger.info(f"User Message: {user_message}")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Embedding Name: {embedding_name}")
    logger.info(f"Response: {answer}")
    logger.info(f"source documents: {formatted_sources}")
    logger.info(f"history: {history}")
    record_interaction_in_database(user_id, 
                                   session_type, 
                                   model_name=model_name, 
                                   embedding_name=embedding_name, 
                                   user_question=user_message,
                                   user_question_ts=ts, 
                                   bot_response=formatted_response, 
                                   bot_response_ts=bot_response_ts,
                                   formatted_sources=formatted_sources,
                                   history=history_string)

def handle_reaction(body, logger):
    logger.info(f"Reaction added event triggered.")
    event = body['event']
    reaction = event['reaction']
    user_id = event['user']
    session_type = 'reaction'
    item = event['item']
    user_reaction_ts = event['event_ts']
    if item['type'] != 'message':
        return
    # only record reaction that is thumbsup or thumbsdown
    if reaction in ['-1', '+1']:
        record_interaction_in_database(user_id,
                                       session_type, 
                                       user_reaction = reaction, 
                                       bot_response_ts=item['ts'],
                                       user_reaction_ts=user_reaction_ts)
    logger.info(f"Recorded reaction: {reaction} from user: {user_id} on message: {item['ts']}")

def activate_safety_mode(ack, command):
    user_id = command["user_id"]
    # Get the chat state for the user
    chat_state = user_chat_states.get(user_id)
    if not chat_state:
        chat_state = ChatState()
        user_chat_states[user_id] = chat_state
    # Update safety_mode
    chat_state.safety_mode = not chat_state.safety_mode
    chat_state.chatbot_instance = None
    safety_mode = chat_state.safety_mode 

    if safety_mode:
        message = "Safety mode activated. The model is set to <blank> models and hugging face embeddings."
    else:
        message = "Safety mode deactivated. You can now select the model and embedding of your choice."
    slack_client.chat_postMessage(channel=user_id, text=message)
    ack()  


def cleanup_inactive_chatbots(timeout):
    current_time = time.time()
    with user_chat_states_lock:
        # Use list() to copy keys to a new list, because we can't modify the dictionary during iteration
        for key in list(user_chat_states.keys()):
            if current_time - user_chat_states[key].last_active > timeout:
                del user_chat_states[key]

def periodic_cleanup():
    cleanup_inactive_chatbots(600)  # Cleanup instances inactive for more than an hour
    Timer(600, periodic_cleanup).start()  # Run cleanup every 10 minutes


def search_messages(query, channels, user_id, slack_user_client):
    """
    let the model search through the channel history (that the app is installed) for the query and retreive info together with
    knowledge base to answer the question. This feature only works in safety mode
    """
    # Get the chat state for the user
    chat_state = user_chat_states.get(user_id)
    if not chat_state or not chat_state.safety_mode:
        # Safety mode not activated, return empty list or raise an error
        return []

    search_message_list = []
    keyword_query = extract_keywords(query)
    for channel in channels:
        channel_query = f'{keyword_query} in:{channel}'
        response = slack_user_client.search_messages(query=channel_query, 
                                                     sort='timestamp', 
                                                     sort_dir='desc', 
                                                     count=5)
        if response["ok"]:
            matches = response["messages"]["matches"]
            for match in matches:
                channel_id = match["channel"]["id"]
                channel_name = match["channel"]["name"]
                permalink = match["permalink"]
                text = match["text"]
                ts = match["ts"]
                search_type = match["type"]
                search_message_list.append({
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "permalink": permalink,
                    "text": text,
                    "ts": ts,
                    "type": search_type
                })
        else:
            raise Exception(f"API request failed: {response['error']}")
    return search_message_list

