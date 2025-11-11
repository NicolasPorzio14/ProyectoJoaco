import streamlit as st
import os

# --- LangChain moderno / ecosistema ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# --- 1. UI ---
st.set_page_config(page_title="TITI-AYUDANTE IMPOSITIVO", layout="wide")
st.header("TITI-AYUDANTE IMPOSITIVO 💰🤖")


# --- 2. API Key ---
def get_openai_api_key():
    """Formulario para ingresar la clave API."""
    with st.sidebar:
        st.markdown("## 🔑 Clave OpenAI")
        input_text = st.text_input(
            label="OpenAI API Key",
            placeholder="Ingresa tu clave sk-...",
            type="password"
        )
        if not input_text:
            st.warning("⚠️ Por favor, ingresa tu clave API para comenzar.")
    return input_text

openai_api_key = get_openai_api_key()


# --- 3. Rutas / paths ---
PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "IVA.pdf")
VECTOR_DB_PATH = "faiss_index_iva"  # opcional si querés persistir


# --- 4. Carga y vectorización del PDF ---
@st.cache_resource
def process_document(api_key: str):
    """
    Carga el PDF, lo parte en chunks, genera embeddings y crea el Vector Store (FAISS).
    """
    if not os.path.exists(PDF_PATH):
        st.error(f"¡ERROR! No se encontró el archivo en: {PDF_PATH}")
        st.stop()

    try:
        # Cargar documento
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        # Partir en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = OpenAIEmbeddings(api_key=api_key)
        # Si preferís especificar modelo de embeddings:
        # embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

        # Vector store
        vectorstore = FAISS.from_documents(texts, embeddings)

        # (Opcional) persistencia:
        # vectorstore.save_local(VECTOR_DB_PATH)

        return vectorstore
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el documento o crear embeddings: {e}")
        st.stop()


# --- 5. Construcción de la cadena RAG moderna ---
def build_rag_chain(vectorstore, api_key: str):
    """
    Arma: history-aware retriever + stuff documents chain + retrieval chain.
    """
    system_prompt_text = (
        "Eres TITI, un Contador Público Profesional de Argentina especializado en derecho tributario y asesoramiento impositivo. "
        "Tu función es responder a las consultas del usuario basándote estricta y exclusivamente en el contexto que te proporciona el archivo 'IVA.pdf' (el contexto recuperado). "
        "Utiliza un lenguaje formal, técnico y profesional, como corresponde a un experto en la materia. "
        "Si la respuesta no se encuentra en el contexto proporcionado, debes responder: "
        "'Lo siento, como Contador Impositivo, solo puedo responder basándome en la información del documento IVA.pdf, y no encontré esa información específica en él.' "
        "No utilices conocimientos generales."
    )

    # Modelo de chat
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # puedes cambiar a "gpt-4o-mini" si lo preferís
        temperature=0.2,
        api_key=api_key
    )

    # Retriever base
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Prompt para reescritura de la consulta (contextualización con historial)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Dada la conversación previa y la última pregunta del usuario, reescribe la pregunta para que sea una consulta autónoma y clara. "
         "No respondas la pregunta; solo reescríbela si hace falta. Si ya es autónoma, deja la consulta tal cual."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Retriever consciente del historial
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )

    # Prompt de respuesta final (usa el contexto recuperado)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        MessagesPlaceholder("chat_history"),
        ("human",
         "Usa exclusivamente el siguiente contexto del documento para responder a la pregunta.\n\n"
         "----------------\n"
         "Contexto del documento:\n{context}\n"
         "----------------\n"
         "Pregunta del Usuario: {input}")
    ])

    # Cadena que combina documentos + prompt (stuff)
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=answer_prompt
    )

    # Cadena RAG final
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain
    )
    return rag_chain


# --- 6. Estado de la app ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # lista de HumanMessage/AIMessage
if "is_setup_done" not in st.session_state:
    st.session_state.is_setup_done = False


def setup_backend():
    """Inicializa vectorstore + cadena RAG una sola vez."""
    if openai_api_key:
        with st.spinner("Procesando documentos y preparando el Ayudante Impositivo..."):
            vectorstore = process_document(openai_api_key)
            st.session_state.rag_chain = build_rag_chain(vectorstore, openai_api_key)
            st.session_state.is_setup_done = True
        st.success("¡TITI-AYUDANTE IMPOSITIVO listo! Ya puedes preguntar.")

if not st.session_state.is_setup_done and openai_api_key:
    setup_backend()
elif not openai_api_key:
    st.info("Ingresa tu clave API en la barra lateral para cargar los datos del IVA.")


# --- 7. Interacción de chat ---
def handle_user_input(user_question: str):
    """Ejecuta la cadena RAG con historial y muestra la respuesta."""
    if st.session_state.rag_chain is None:
        st.error("El modelo aún no está configurado. Verifica la clave API y la carga inicial.")
        return

    try:
        result = st.session_state.rag_chain.invoke({
            "input": user_question,
            "chat_history": st.session_state.chat_history
        })
        answer = result.get("answer", "")

        # Actualizar historial de conversación
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=answer))

    except Exception as e:
        st.error(f"Error al obtener respuesta del LLM: {e}")


# --- 8. UI de entrada y render del historial ---
user_question = st.chat_input("Pregunta algo sobre el archivo IVA.pdf...")

if user_question and st.session_state.is_setup_done:
    with st.spinner("TITI está pensando..."):
        handle_user_input(user_question)
elif user_question and not st.session_state.is_setup_done:
    st.warning("El Ayudante Impositivo no está listo. Verifica que la clave API esté ingresada y que el PDF se haya cargado correctamente.")

st.markdown("### 💬 Historial de Conversación")
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
