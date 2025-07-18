import streamlit as st
from openai import OpenAI
from io import BytesIO
import json
import base64
import openai
from docx import Document
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

openai_client = st.secrets["OPENAI_API_KEY"]

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
QDRANT_COLLECTION_NAME = "notes"

# --- FUNKCJE ---

# Funkcja: transkrypcja
def transcribe_audio_to_text(audio_path):
    with open(audio_path, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="text"
        )
    return transcript

# Funkcja: klasyfikacja na role
def classify_speakers(transcript_text):
    prompt = (
        f"""Podziel poniższą transkrypcję rozmowy na role: Doradca i Klient.

        Założenia:
        - Nie wymyślaj nowych wypowiedzi – trzymaj się dokładnie tego, co jest w transkrypcji.
        - Przypisuj role logicznie, na podstawie stylu wypowiedzi i treści.
        - Jeśli nie masz pewności – oznacz wypowiedź jako „[Nieznane]”.
        - Zwróć tylko tekst dialogu w formacie:
        
        Oto transkrypcja:
        {transcript_text}
        """
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Możesz zmienić na "gpt-3.5-turbo" jeśli potrzebujesz
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

#
# --- BAZA DANYCH ---
#

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path="qdrant_data")

def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print("Tworzę kolekcję")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        print("Kolekcja już istnieje")

def get_embedding(text):
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )

    return result.data[0].embedding

def load_articles_to_qdrant(json_path="articles.json"):
    qdrant_client = get_qdrant_client()

    # Sprawdź, ile punktów już jest w kolekcji
    existing_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True,
    ).count

    # Jeśli już są artykuły – nie ładuj ponownie
    if existing_count > 0:
        print("Artykuły już są w bazie – pomijam ładowanie.")
        return

    # Jeśli kolekcja jest pusta – załaduj artykuły z pliku
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    for i, article in enumerate(articles):
        embedding = get_embedding(f"{article['title']} {article['content']}")
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "id": article["id"],
                        "title": article["title"],
                        "content": article["content"],
                    },
                )
            ]
        )

    print(f"Załadowano {len(articles)} artykułów do Qdrant.")



def search_articles(query, top_k=2):
    qdrant_client = get_qdrant_client()
    embedding = get_embedding(query)

    results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k,
    )

    top_articles = {}
    for i, r in enumerate(results):
        top_articles[f"article{i+1}"] = {
            "title": r.payload.get("title", ""),
            "content": r.payload.get("content", "")
        }

    return top_articles

# Qdrant initialization
assure_db_collection_exists()
load_articles_to_qdrant("articles.json")

# Funkcja generująca podsumowanie
def article_creation(edited_text, top_articles, temp_lvl):
    article1 = top_articles.get("article1", {"title": "", "content": ""})
    article2 = top_articles.get("article2", {"title": "", "content": ""})
    prompt = (
        f"""
        Na podstawie poniższej transkrypcji rozmowy z klientem oraz dwóch artykułów, napisz nowy artykuł reklamowo-informacyjny do gazety:

        - Ma on być atrakcyjny i przekonujący dla klienta z rozmowy.
        - Uwzględnij potrzeby, problemy lub pytania, które pojawiły się w transkrypcji.
        - Wykorzystaj treści i argumenty z podanych artykułów, ale nie kopiuj ich dosłownie.
        - Artykuł ma mieć długość zbliżoną do średniej liczby słów z obu artykułów.
        - Napisz tekst w stylu poradnikowym lub storytellingowym – dostosuj ton do tego, co wynika z rozmowy.

        ---

        📞 Rozmowa z klientem:
        \"\"\"{edited_text}\"\"\"

        ---

        📰 Artykuł 1:
        Tytuł: {article1['title']}
        Treść: {article1['content']}

        📰 Artykuł 2:
        Tytuł: {article2['title']}
        Treść: {article2['content']}

        ---

        ✍️ Wygeneruj nowy artykuł reklamowy na bazie powyższych danych.
        """
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temp_lvl
    )
    return response.choices[0].message.content.strip()

# Funkcja: eksport do DOCX
def save_text_to_docx(text: str) -> BytesIO:
    doc = Document()
    doc.add_paragraph(text)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Session state initialization
    
if "final_article" not in st.session_state:
    st.session_state['final_article'] = ""

if "uploaded_file" not in st.session_state:
    st.session_state['uploaded_file'] = ""

if 'edited_text' not in st.session_state:
    st.session_state['edited_text'] = ""

#
# --- STREAMLIT UI ---
#
st.set_page_config(page_title="Transkryptor", layout="wide")
st.title("🎙️ Audio2Doc — Transkrypcja z przypisaniem ról")
st.session_state['uploaded_file'] = st.file_uploader("Wgraj rozmowę z klientem", type=["mp3"])

if st.session_state['uploaded_file']:
    st.audio(st.session_state['uploaded_file'], format='audio/mp3')

    with st.spinner("Ładuję plik"):
        with open("temp.mp3", "wb") as f:
            f.write(st.session_state['uploaded_file'].read())
        

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("📄 Transkrybuj i dziel na role"):
            with st.spinner("Transkrybuję"):
                transcript = transcribe_audio_to_text("temp.mp3")
            with st.spinner("Dzielę na role"):
                st.session_state['edited_text'] = classify_speakers(transcript)
    
        if "edited_text" in st.session_state:
                st.subheader('Transkrypcja')
                st.session_state['edited_text'] = st.text_area("", value=st.session_state['edited_text'], height=600)


    with col2:
        value = st.slider('Creativity', 0, 100, 30)
        temp_level = (value/100)
        if st.button('☢️'):
            with st.spinner("Tworzę artykuł ⚔️"):
                top_articles = search_articles(st.session_state['edited_text'])
                st.session_state['final_article'] = article_creation(st.session_state['edited_text'], top_articles, temp_level)

        if "final_article" in st.session_state:
            st.session_state['final_article'] = st.text_area('Artykuł:', st.session_state['final_article'], height=600)


    final_text = f"{st.session_state['edited_text']}\n\n{st.session_state['final_article']}"
    docx_buffer = save_text_to_docx(final_text)
    b64 = base64.b64encode(docx_buffer.getvalue()).decode()

    button_html = f"""
        <div style="text-align: center; margin-top: 30px;">
            <a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}"
            download="transkrypcja_z_rolami.docx"
            style="
                background-color: #4CAF50;
                color: white;
                padding: 16px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 18px;
                border-radius: 8px;
                font-weight: bold;
            ">
            📥 Pobierz jako .docx
            </a>
        </div>
    """

    # Wyświetl niestandardowy przycisk
    st.markdown(button_html, unsafe_allow_html=True)
