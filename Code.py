import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
import faiss
from google import genai
from google.genai import types
import os

# ---------------------
# CONFIG
# ---------------------
os.environ["GEMINI_API_KEY"] = "AIzaSyDeBf2DTHQy4bZjWNMEPftPpJc5G6DZB9w"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
LLM_MODEL = "gemini-3-flash-preview"

# ---------------------
# FUNCTIONS
# ---------------------

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_text(texts):
    # MOCK embeddings to avoid Gemini quota issues
    return np.random.rand(len(texts), 768).astype("float32")
    """
    # If you have quota, uncomment real embedding:
    embeddings = []
    for t in texts:
        content = [{"role": "user", "text": t}]
        response = client.models.embed_content(
            model="models/embedding-001",
            contents=content
        )
        embeddings.append(response.embeddings[0].values)
    return np.array(embeddings).astype("float32")
    """

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_chunks(query, index, chunks, k=3):
    # MOCK retrieval: for testing
    return chunks[:3]  # just return first 3 chunks
    """
    # Real version:
    query_embedding = embed_text([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]
    """



def ask_gemini(context, question):
    prompt = f"""
You are an AI tutor.
Answer ONLY using the context below.
If the answer is not in the context, say "Not found in the document."

Context:
{context}

Question:
{question}
"""
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]
    
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=contents
    )

    
    return response.candidates[0].content.parts[0].text
  # or response.text depending on SDK version



# ---------------------
# STREAMLIT UI
# ---------------------

st.title("AI Tutor")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    text = read_pdf(uploaded_file)
    chunks = chunk_text(text)

    embeddings = embed_text(chunks)
    index = create_faiss_index(embeddings)

    st.success("Document processed and indexed")

    question = st.text_input("Ask a question")

    if question:
        retrieved_chunks = retrieve_chunks(question, index, chunks)
        context = "\n".join(retrieved_chunks)
        answer = ask_gemini(context, question)
        st.write("### Answer")
        st.write(answer)
