import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from langchain.memory import ConversationBufferMemory  # (seguimos usando para conveniencia en UI)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# NUEVO: cadenas RAG de la API moderna de LangChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. CONFIGURACIÓN INICIAL DEL FRONTEND ---

st.set_page_config(page_title="TITI-AYUDANTE IMPOSITIVO", layout="wide")
st.header("TITI-AYUDANTE IMPOSITIVO 💰🤖")

# --- 2. GESTIÓN DE LA CLAVE API Y LA LÓGICA DE BACKEND ---

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

# Ruta estática al archivo PDF
PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "IVA.pdf")  # corregido __file__
# La base de datos vectorial se guarda en el mismo directorio que el script
VECTOR_DB_PATH = "faiss_index_iva"

# Función para cargar y procesar el documento
@st.cache_resource
def process_document(api_key: str):
    """
    Carga el PDF, aplica splits, embeddings y crea/recupera el Vector Store.
    """
    if not os.path.exists(PDF_PATH):
        st.error(f"¡ERROR! No se encontró el archivo en: {PDF_PATH}")
        st.stop()

    try:
        # Cargar documento
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        # Aplicar splits
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)

        # Crear embeddings
        embeddings = OpenAIEmbeddings(api_key=api_key)

        # Si querés persistir, podrías usar FAISS.save_local / load_local
        vectorstore = FAISS.from_documents(texts, embeddings)

        return vectorstore
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el documento o crear embeddings: {e}")
        st.stop()


# --- 2.b PROMPTS Y CADENAS RAG MODERNAS ---

def build_rag_chain(vectorstore, api_key: str):
    """
    Crea un retriever con consciencia de historial y una cadena RAG moderna:
    history-aware retriever + stuff documents chain + retrieval chain.
    """
    system_prompt_text = (
        "Eres TITI, un Contador Público Profesional de Argentina especializado en derecho tributario y asesoramiento impositivo. "
        "Tu función es responder a las consultas del usuario basándote estricta y exclusivamente en el contexto que te proporciona el archivo 'IVA.pdf' (el contexto recuperado). "
        "Utiliza un lenguaje formal, técnico y profesional, como corresponde a un experto en la materia. "
        "Si la respuesta no se encuentra en el contexto proporcionado, debes responder: "
        "'Lo siento, como Contador Impositivo, solo puedo responder basándome en la información del documento IVA.pdf, y no encontré esa información específica en él.' "
        "No utilices conocimientos generales."
    )

    # Modelo
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        api_key=api_key
    )

    # Retriever base
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Prompt para reescritura de consulta según historial (query rewriter)
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

    # Cadena para combinar documentos (stuff) y generar la respuesta
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=answer_prompt
    )

    # Cadena RAG final: (history-aware retriever) -> (document_chain)
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain
    )
    return rag_chain


# --- 3. LÓGICA DE CHAT Y ESTADO ---

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # lista de LangChain Messages
if "is_setup_done" not in st.session_state:
    st.session_state.is_setup_done = False

def setup_backend():
    """Función que inicia el backend (solo se llama una vez)."""
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

def handle_user_input(user_question: str):
    """Maneja la pregunta del usuario usando la cadena RAG moderna."""
    if st.session_state.rag_chain is None:
        st.error("El modelo aún no está configurado. Por favor, verifica tu clave API y espera la carga inicial.")
        return

    try:
        # invocamos pasando el historial y la entrada actual
        result = st.session_state.rag_chain.invoke({
            "input": user_question,
            "chat_history": st.session_state.chat_history
        })
        answer = result.get("answer", "")

        # Actualizamos historial (LangChain Messages)
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=answer))

    except Exception as e:
        st.error(f"Error al obtener respuesta del LLM: {e}")
        return

# --- 4. INTERFAZ DE CONVERSACIÓN ---

user_question = st.chat_input("Pregunta algo sobre el archivo IVA.pdf...")

if user_question and st.session_state.is_setup_done:
    with st.spinner("TITI está pensando..."):
        handle_user_input(user_question)
elif user_question and not st.session_state.is_setup_done:
    st.warning("El Ayudante Impositivo no está listo. Verifica que la clave API esté ingresada y que el PDF se haya cargado correctamente.")

# Mostrar historial de chat
st.markdown("### 💬 Historial de Conversación")

if st.session_state.chat_history:
    # Render simple según tipo de mensaje
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
