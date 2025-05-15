from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pinecone import Pinecone
from pypdf import PdfReader
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv

load_dotenv()
pine_api_key = os.getenv("PINECONE_API_KEY")

# **********Functions to help you load documents to PINECONE************


# Read PDF data
def read_pdf_data(pdf_file):
    pdf_page = PdfReader(pdf_file)
    text = ""
    for page in pdf_page.pages:
        text += page.extract_text()
    return text


# Split data into chunks
def split_data(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    docs = splitter.split_text(text)
    doc_chunks = splitter.create_documents(docs)
    return doc_chunks


# Create embedding model
def get_embeddings():
    return SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Push documents and embeddings to Pinecone
def push_to_pine(embeddings, docs):
    pine = Pinecone(api_key=pine_api_key)
    vectorstore = PineconeVectorStore(index=pine.Index("ticket"), embedding=embeddings)
    vectorstore.add_documents(documents=docs)
    return vectorstore


# *********Functions for dealing with Model related tasks...************


# 1. Read CSV and assign clear column names
def read_data(data):
    df = pd.read_csv(data, header=None, names=["text", "label"])
    return df


# 2. Generate embeddings for the 'text' column
def create_embeddings(df, embeddings):
    df["embedding"] = df["text"].apply(lambda x: embeddings.embed_query(x))
    return df


# 3. Split into train/test using 'embedding' and 'label'
def split_train_test_data(df_sample):
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(
        list(df_sample["embedding"]),
        list(df_sample["label"]),
        test_size=0.25,
        random_state=0,
    )
    print("Training samples:", len(sentences_train))
    return sentences_train, sentences_test, labels_train, labels_test


# 4. Accuracy score
def get_score(svm_classifier, sentences_test, labels_test):
    score = svm_classifier.score(sentences_test, labels_test)
    return score
