from nltk.tokenize import word_tokenize
from constants import important_keywords
import re

def parse_model_and_embedding(user_message, say, ts, available_models, available_embeddings):
    #parse the model command from the user text input
    model_match = re.search(r'/model\s*([^ ]*)', user_message)
    embedding_match = re.search(r'/embedding\s*([^ ]*)', user_message)
    if model_match is not None and model_match.group(1) in available_models:
        model_name = model_match.group(1)
    else:
        say(text="Warning: Specified model not found, using default model: gpt-3.5-turbo", thread_ts=ts)
        model_name = "gpt-3.5-turbo"

    if embedding_match is not None and embedding_match.group(1) in available_embeddings:
        embedding_name = embedding_match.group(1)
    else:
        say(text="Warning: Specified embedding not found, using default embedding: openai-embedding", thread_ts=ts)
        embedding_name = "openai-embedding"
    
    return model_name, embedding_name


def get_model_and_embedding(safety_mode, user_message, say, ts):
    """
    return the model and embbedding based on safety mode and user message

    Args:
        safety_mode (bool): The current status of the safety mode
        user_message (str): The user's message.
        say (callable): A function for sending a response to the channel.
        ts (str): The timestamp of the user's message.
    Returns:
        (str, str): The model and embedding names.
    """
    if safety_mode:
        return "<intentionally left blank>", "hf-sentence-transformer"
    else:
        return parse_model_and_embedding(user_message, say, ts)
    

def clean_user_message(user_message):
    """
    Clean the user's message by removing the model and embedding commands.

    Args:
        user_message (str): The user's message.
    Returns:
        (str): The cleaned user message.
    """
    user_message = re.sub(r'/model\s*([^ ]*)', '', user_message)
    user_message = re.sub(r'/embedding\s*([^ ]*)', '', user_message)
    return user_message


def get_lazy_model_and_embedding(model_name, embedding_name, models, embeddings):
    lazy_model = models.get(model_name)
    lazy_embedding = embeddings.get(embedding_name)
    return lazy_model, lazy_embedding

def get_response_and_history(qa, user_message, history, tokenizer):
    response = qa({"question": user_message, "chat_history": history})
    answer = response['answer']
    new_history_item = (user_message, response["answer"])
    new_history_item_tokens = len(tokenizer.encode(' '.join(new_history_item)))
    while len(history) > 0 and sum(len(tokenizer.encode(' '.join(item))) for item in history) + new_history_item_tokens > TOKEN_LIMIT:
        history.pop(0)
    history.append(new_history_item)
    return response, answer, history

def format_response(response, answer):
    metadata_list = [doc.metadata for doc in response["source_documents"]]
    # Remove duplicate sources
    unique_sources = list(set([meta['source'] for meta in metadata_list]))
    # Combine the answer and metadata list into a single string
    formatted_sources = '\n'.join([f"source: {src}" for src in unique_sources])
    # Combine the answer and sources into a single string
    formatted_response = f"{answer}\n\n{formatted_sources}"
    return formatted_response, formatted_sources

def extract_keywords(query):
    """
    extract keywords from the query
    """
    word_tokens = word_tokenize(query.lower())
    # Filter out only important keywords
    filtered_words = [w for w in word_tokens if w in important_keywords]
    return " ".join(filtered_words)
    
   
