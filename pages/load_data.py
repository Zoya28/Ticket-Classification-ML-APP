import streamlit as st
from dotenv import load_dotenv
from pages.admin_backend import *


def main():
    load_dotenv()
    st.set_page_config(page_title="Dump PDF to Pinecone - Vector Store")
    st.title("Please upload your files...📁 ")

    # Upload the pdf file...
    pdf = st.file_uploader("Only PDF files allowed", type=["pdf"])

    if pdf is not None:
        with st.spinner("Wait for it..."):
            text = read_pdf_data(pdf)
        st.success("👉Reading PDF done")

        with st.spinner("Splitting data into chunks..."):
            chunks = split_data(text)
        st.success("Data chunked! 🧠")

        with st.spinner("Creating embeddings..."):
            embeddings = get_embeddings()
        st.success("Embeddings created! 🔐")

        with st.spinner("Pushing data to Pinecone vault..."):
            push_to_pine(embeddings=embeddings, docs=chunks)
        st.sidebar.success(
            "All set! Successfully pushed the embeddings to Pinecone. 🎉"
        )


if __name__ == "__main__":
    main()
