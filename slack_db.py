import sqlite3

def setup_database():
    conn = sqlite3.connect('slackbot.db')  
    c = conn.cursor()

    # Create table
    c.execute('''
        CREATE TABLE IF NOT EXISTS interactions
        (user_id text, session_type text, model_name text, embedding_name text, user_question text, user_question_ts text, bot_response text, bot_response_ts text, user_reaction text, user_reaction_ts text, formatted_sources text, history text)
    ''')

    conn.commit()  
    conn.close()  

def record_interaction_in_database(user_id, 
                                   session_type, 
                                   model_name=None, 
                                   embedding_name=None, 
                                   user_question=None,
                                   user_question_ts=None,
                                   bot_response=None, 
                                   bot_response_ts=None, 
                                   user_reaction=None,
                                   user_reaction_ts=None, 
                                   formatted_sources=None,
                                   history=None):
    conn = sqlite3.connect('slackbot.db')
    c = conn.cursor()

    c.execute('''
        INSERT INTO interactions (user_id, session_type, model_name, embedding_name, user_question, user_question_ts, bot_response, bot_response_ts, user_reaction, user_reaction_ts, formatted_sources, history)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, session_type, model_name, embedding_name, user_question, user_question_ts, bot_response, bot_response_ts, user_reaction, user_reaction_ts, formatted_sources, history))

    conn.commit()  
    conn.close()  
