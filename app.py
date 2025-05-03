import streamlit as st
import hashlib
import tempfile
import os
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader  # For PDF text extraction (fallback)
from openai import OpenAI  # For GPT API calls
import fitz  # PyMuPDF

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Streamlit UI
st.set_page_config(page_title="Assistant - Sécurité Routière", layout="centered")
st.title("📄 Posez des questions sur la sécurité Routière")

# Default system message
default_system_msg = """**Expert Sécurité Routière Strict**  

*Mission* : Répondre EXCLUSIVEMENT aux questions sur la sécurité routière avec une expertise maximale.  

**Règles Absolues** :  
1. **Filtre Thématique** :  
   - Si la question ne concerne PAS la sécurité routière et code de la route → Réponse systématique :  
     *"Désolé, je suis strictement limité aux questions de sécurité routière. Je ne peux pas répondre à cette demande."*  
   - Aucune exception (même pour les questions connexes comme mécanique ou GPS)  

2. **Priorité Documentaire** :  
   - "Le document indique que..." pour toute information sourcée  
   - Croiser les informations du document avec le Code de la Route officiel  
   - Marquer clairement : [SOURCE DOCUMENT] / [CONNAISSANCE EXPERTE]  

3. **Précision Technique** :  
   - Chiffres exacts (ex : distances de freinage, taux d'alcoolémie)  
   - Citations précises des articles du Code de la Route quand applicable  
   - Mises à jour législatives récentes  

4. **Format de Réponse** :  
   ▶ **Réponse Directe** : 1-2 phrases maximum  
   ▶ **Explications** : Bullet points techniques si nécessaire  
   ▶ **Sources** :  
       - Document fourni (page X si disponible)  
       - Article R.XXX du Code de la Route  
   ▶ **Avertissements** : "Vérifiez toujours les textes officiels"  

**Interdictions** :  
- Pas de hors-sujet  
- Pas de conseils personnalisés (ex : "Dans votre cas...")  
- Pas d'interprétation libre des lois"""

# File to load
pdf_path = "nosia.pdf"

# Session state setup
if 'system_msg' not in st.session_state:
    st.session_state.system_msg = default_system_msg

# PDF processing
@st.cache_resource(show_spinner="Chargement du modèle d'embedding...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@st.cache_data(show_spinner="Analyse du PDF en cours...")
def process_pdf(file_path: str) -> Tuple[List[str], np.ndarray]:
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(text)
    embedder = load_embedding_model()
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return chunks, embeddings

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        if text.strip():
            return text
    except:
        pass
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    try:
        reader = PdfReader(tmp_path)
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        return text if text.strip() else ""
    finally:
        os.unlink(tmp_path)

def chunk_text(text: str, chunk_size=512, overlap=64) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def search_chunks(query: str, chunks: List[str], embeddings: np.ndarray, k=5) -> List[Tuple[str, float]]:
    embedder = load_embedding_model()
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [(chunks[i], float(similarities[i])) for i in top_indices if i < len(chunks)]

def generate_response(query: str, relevant_chunks: List[Tuple[str, float]], system_msg: str, context: str):
    prompt = f"""QUESTION DE L'UTILISATEUR : {query}

{context}

ANALYSE ET RÉPONSE :
1. Cette question concerne-t-elle le document ou la réglementation routière ?
2. Quelles informations pertinentes existent dans le document ?
3. Quelles connaissances complémentaires peuvent être ajoutées avec prudence ?
4. Formulez la réponse la plus précise et sécuritaire."""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

# Load and cache chunks/embeddings
if os.path.exists(pdf_path) and 'chunks' not in st.session_state:
    chunks, embeddings = process_pdf(pdf_path)
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings

# Form input
with st.form(key='query_form'):
    # Full-width input for system message
    system_msg_input = st.text_area(
        "Modifiez le message du système (règles pour l'Assistant)",
        value=st.session_state.system_msg,
        height=200,
        key="system_msg_input"
    )

    if st.form_submit_button("Afficher le message par défaut"):
        st.info("Message système par défaut :")
        st.code(default_system_msg, language="markdown")

    # Question input and submit
    query = st.text_input("Posez votre question sur le document:")
    submitted = st.form_submit_button("Soumettre la question")

# Handle query submission
if submitted and query:
    with st.spinner("Recherche des réponses..."):
        relevant_chunks = search_chunks(query, st.session_state.chunks, st.session_state.embeddings)

        if not relevant_chunks:
            st.warning("Aucune information pertinente trouvée dans le document.")
        else:
            # Context is now dynamically generated from chunks
            context = "\n\n---\n\n".join([
                f"EXTRAIT DOCUMENT (Pertinence: {score:.2f}):\n{text}"
                for text, score in relevant_chunks
            ])

            # Update session system message
            st.session_state.system_msg = system_msg_input

            answer = generate_response(query, relevant_chunks, st.session_state.system_msg, context)
            st.success(answer)

            with st.expander("Voir les passages pertinents"):
                for chunk, score in relevant_chunks:
                    st.write(f"Pertinence: {score:.2f}")
                    st.text(chunk)
                    st.divider()
