from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS


def initialize_chatbot(lazy_model, lazy_embedding, config):
    
    model = lazy_model.model
    embeddings = lazy_embedding.embedding
    embedding_vb_name = lazy_embedding.config['vb_name']
    vb_config = next(vb for vb in config['vectordb'] if vb['vb_name'] == embedding_vb_name)
    faiss_path = vb_config['path']
    # load faiss index
    db = FAISS.load_local(faiss_path, embeddings)
    #initialize conversation retreival chain
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(model, retriever, verbose=True, return_source_documents=True)
    #print(f'Chatbot initialized: {qa}')
    return qa