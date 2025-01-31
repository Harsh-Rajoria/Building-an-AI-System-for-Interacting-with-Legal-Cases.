import os
import re
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

genai.configure(api_key="AIzaSyDMPwiBmm3G2QCwCA-XmFmKUXHvaziqurM")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def save_uploaded_files(uploaded_files, upload_folder):
    os.makedirs(upload_folder, exist_ok=True)
    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
    return saved_files

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def similarity_search(query, index, model, metadata, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx != -1:
            filename, chunk = metadata[idx]
            results.append({"filename": filename, "chunk": chunk, "distance": distance})
    return results

def generate_response(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def main():
    st.set_page_config(page_title="LexPilot", page_icon="⚖️", layout="wide")

    # Initialize session state for navigation
    if "current_step" not in st.session_state:
        st.session_state.current_step = "Upload PDFs"

    st.sidebar.title("Navigation")
    st.sidebar.write("Current Step: **{}**".format(st.session_state.current_step))

    # Add a "Next" button to move to the next step
    if st.session_state.current_step == "Upload PDFs":
        if st.sidebar.button("Next: Process PDFs"):
            st.session_state.current_step = "Process PDFs"
            st.rerun()
    elif st.session_state.current_step == "Process PDFs":
        if st.sidebar.button("Next: Search & Generate"):
            st.session_state.current_step = "Search & Generate"
            st.rerun()
    elif st.session_state.current_step == "Search & Generate":
        if st.sidebar.button("Restart: Upload PDFs"):
            st.session_state.current_step = "Upload PDFs"
            st.rerun()

    if st.session_state.current_step == "Upload PDFs":
        st.title("📤 Upload PDFs")
        st.write("Upload your legal documents in PDF format.")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            upload_folder = "uploaded_files"
            saved_files = save_uploaded_files(uploaded_files, upload_folder)
            st.success(f"Uploaded {len(saved_files)} files successfully!")
            st.write("### Uploaded Files:")
            for file_path in saved_files:
                st.write(f"- {os.path.basename(file_path)}")

    elif st.session_state.current_step == "Process PDFs":
        st.title("🔧 Process PDFs")
        st.write("Extract, clean, chunk, and vectorize the text from uploaded PDFs.")

        if st.button("Start Processing"):
            upload_folder = "uploaded_files"
            output_folder = "cleaned_texts"
            chunk_folder = "text_chunks"
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(chunk_folder, exist_ok=True)

            saved_files = [os.path.join(upload_folder, f) for f in os.listdir(upload_folder) if f.endswith(".pdf")]
            if not saved_files:
                st.warning("No PDFs found. Please upload files first.")
            else:
                all_chunks = []
                all_filenames = []

                progress_bar = st.progress(0)
                for i, file_path in enumerate(saved_files):
                    st.write(f"Processing: {os.path.basename(file_path)}")
                    text = extract_text_from_pdf(file_path)
                    cleaned_text = clean_text(text)
                    save_cleaned_text(os.path.basename(file_path), cleaned_text, output_folder)
                    chunks = split_text_into_chunks(cleaned_text)
                    save_chunks_to_files(os.path.basename(file_path), chunks, chunk_folder)
                    all_chunks.extend(chunks)
                    all_filenames.extend([os.path.basename(file_path)] * len(chunks))
                    progress_bar.progress((i + 1) / len(saved_files))

                st.write("### Generating Embeddings...")
                model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = generate_embeddings(all_chunks)
                st.write(f"**Total Chunks:** {len(all_chunks)}")
                st.write(f"**Embedding Dimension:** {embeddings.shape[1]}")

                st.write("### Creating FAISS Index...")
                index = create_faiss_index(embeddings)
                st.success("FAISS index created successfully.")

                faiss.write_index(index, "faiss_index.index")
                metadata = list(zip(all_filenames, all_chunks))
                with open("faiss_index_metadata.txt", "w") as f:
                    for filename, chunk in metadata:
                        f.write(f"{filename}\t{chunk}\n")
                st.success("FAISS index and metadata saved.")

                st.session_state.metadata = metadata
                st.session_state.model = model
                st.session_state.index = index

    elif st.session_state.current_step == "Search & Generate":
        st.title("🔍 Search & Generate")
        st.write("Search for relevant information and generate responses using the LLM.")

        if "index" not in st.session_state:
            st.warning("Please process the PDFs first.")
        else:
            query = st.text_input("Enter your query:")
            if query:
                results = similarity_search(query, st.session_state.index, st.session_state.model, st.session_state.metadata)
                st.write("### Search Results:")
                for result in results:
                    with st.expander(f"**File:** {result['filename']} | **Distance:** {result['distance']:.4f}"):
                        st.write(result["chunk"])

                context = "\n\n".join([result["chunk"] for result in results])
                with st.expander("### Combined Context for LLM"):
                    st.write(context)

                if st.button("Generate Response"):
                    response = generate_response(query, context)
                    st.write("### LLM Response:")
                    st.write(response)

def save_cleaned_text(filename, text, output_folder):
    output_path = os.path.join(output_folder, filename.replace(".pdf", ".txt"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def save_chunks_to_files(filename, chunks, chunk_folder):
    base_filename = filename.replace(".pdf", "")
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{base_filename}_chunk_{i+1}.txt"
        chunk_path = os.path.join(chunk_folder, chunk_filename)
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(chunk)

if __name__ == "__main__":
    main()
