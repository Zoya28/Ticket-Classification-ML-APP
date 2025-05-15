from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import joblib
load_dotenv()

pine_api_key = os.getenv("PINECONE_API_KEY")


# Create embedding model
def create_embeddings():
    return SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Retrieve existing Pinecone index
def pull_to_pine(embeddings, api_key=pine_api_key):
    pine = Pinecone(api_key=api_key)
    vectorstore = PineconeVectorStore(index=pine.Index("ticket"), embedding=embeddings)
    vectorstore.from_existing_index(index_name="ticket", embedding=embeddings)
    return vectorstore


# Perform similarity search
def get_answer(vectorstore, user_input):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.9,
    )
    # Create retriever from docs
    retriever = vectorstore.as_retriever()

    # Set up the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff"
    )
    # Run the chain
    response = qa_chain.run(user_input)
    return response


def predict(query_result):
    Fitmodel = joblib.load("modelsvm.pk1")
    result = Fitmodel.predict([query_result])
    return result[0]
