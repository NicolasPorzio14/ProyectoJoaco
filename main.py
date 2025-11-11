import streamlit as st
import os

# --- Importaciones Corregidas para LangChain 0.2.x ---
# LangChain Community
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# LangChain OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# LangChain Core/Base
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

# ... líneas anteriores
from langchain_core.runnables.history import RunnableWithMessageHistory

# Componentes LCEL para RAG
# ¡CORRECCIÓN AQUÍ! Se usa el nombre del paquete modular instalado:
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.memory import StreamlitChatMessageHistory # Memoria específica para Streamlit

# --- 1. CONFIGURACIÓN INICIAL DEL FRONTEND ---

st.set_page_config(page_title="TITI-AYUDANTE IMPOSITIVO", layout="wide")
st.header("TITI-AYUDANTE IMPOSITIVO 💰🤖 (LangChain LCEL)")

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

# Asumiendo la estructura: script/data/IVA.pdf
PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "IVA.pdf")

# Función para cargar y procesar el documento (cacheada)
@st.cache_resource
def process_document(api_key: str):
    """
    Carga el PDF, aplica splits, embeddings y crea el Vector Store.
    """
    if not os.path.exists(PDF_PATH):
        st.error(f"¡ERROR! No se encontró el archivo en: {PDF_PATH}")
        st.stop()
        
    try:
        # Cargar documento
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        
        # Aplicar splits (Importación corregida a langchain.text_splitter)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Crear embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Crear la base de datos vectorial (Vector Store)
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el documento o crear embeddings: {e}")
        st.stop()


# Función para inicializar la cadena RAG con memoria (¡Implementación LCEL!)
@st.cache_resource(hash_funcs={FAISS: lambda _: None})
def get_conversation_chain(vectorstore, api_key):
    """
    Crea y retorna la cadena de conversación RAG usando LCEL.
    """
    
    # 1. Definición del Prompt de Sistema
    SYSTEM_PROMPT = (
        "Eres TITI, un Contador Público Profesional de Argentina especializado en derecho tributario y asesoramiento impositivo. "
        "Tu función es responder a las consultas del usuario basándote estricta y exclusivamente en el contexto proporcionado: {context}. "
        "Utiliza un lenguaje formal, técnico y profesional, como corresponde a un experto en la materia. "
        "Si la respuesta no se encuentra en el contexto proporcionado, debes responder: 'Lo siento, como Contador Impositivo, solo puedo responder basándome en la información del documento IVA.pdf, y no encontré esa información específica en él.' "
        "No utilices conocimientos generales."
    )
    
    # 2. Inicializar el LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        openai_api_key=api_key
    )
    
    # --- Componente A: History-Aware Retriever (Reformula la pregunta con contexto) ---
    
    # Prompt para reformular la pregunta (necesario para historial)
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "Dado el historial de chat y la última pregunta del usuario, "
             "formula una pregunta independiente que pueda ser utilizada para la búsqueda en la base de datos vectorial."
            ),
            # La historia de chat será inyectada automáticamente por RunnableWithMessageHistory
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        prompt=contextualize_q_prompt,
    )
    
    # --- Componente B: Stuff Documents Chain (Responde con el contexto y rol) ---
    
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                # El historial se maneja en el paso de recuperación (history_aware_retriever)
                ("human", "{input}"),
            ]
        ),
    )
    
    # --- Componente C: Retrieval Chain (Combina A y B) ---
    
    # Cadena RAG que usa el retriever que maneja el historial y la cadena de documentos
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )
    
    # --- Componente D: Memoria y Conversación Final ---

    # Envolvemos la cadena RAG en un Runnable que maneja el historial con StreamlitChatMessageHistory
    conversation_chain = RunnableWithMessageHistory(
        retrieval_chain,
        lambda session_id: st.session_state.chat_history, # Usa la memoria de Streamlit
        input_messages_key="input",
        history_messages_key="chat_history", # Clave para el historial dentro del prompt
    )
    
    return conversation_chain

# --- 3. LÓGICA DE CHAT Y ESTADO ---

# Inicialización del estado de sesión
if "chat_history" not in st.session_state:
    # Usamos la clase StreamlitChatMessageHistory (de community) para gestionar la memoria
    st.session_state.chat_history = StreamlitChatMessageHistory(key="chat_history") 
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "is_setup_done" not in st.session_state:
    st.session_state.is_setup_done = False


def setup_backend():
    """Función que inicia el backend (solo se llama una vez)."""
    if openai_api_key:
        with st.spinner("Procesando documentos y preparando el Ayudante Impositivo..."):
            # 1. Procesar el documento (obtiene el vector store)
            vectorstore = process_document(openai_api_key)
            
            # 2. Inicializar la cadena de conversación
            st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
            
            # Marcar como listo
            st.session_state.is_setup_done = True
        st.success("¡TITI-AYUDANTE IMPOSITIVO listo! Ya puedes preguntar.")

if not st.session_state.is_setup_done and openai_api_key:
    setup_backend()
elif not openai_api_key:
    st.info("Ingresa tu clave API en la barra lateral para cargar los datos del IVA.")


def handle_user_input(user_question):
    """Maneja la pregunta del usuario y llama al RAG."""
    if st.session_state.conversation is None:
        st.error("El modelo aún no está configurado.")
        return

    # Limpiamos el historial de Streamlit antes de la invocación para evitar duplicados
    # y usamos la memoria gestionada por RunnableWithMessageHistory
    
    try:
        # Llamar a la cadena de conversación RAG (LCEL)
        # Session ID es necesario para el RunnableWithMessageHistory, usamos un placeholder.
        response = st.session_state.conversation.invoke(
            {'input': user_question},
            config={'configurable': {'session_id': 'unique_user_session'}}
        )
        # La respuesta ya se añade automáticamente al historial de Streamlit por la memoria
        return response['answer']
        
    except Exception as e:
        st.error(f"Error al obtener respuesta del LLM: {e}") 
        return None


# --- 4. INTERFAZ DE CONVERSACIÓN ---

# Mostrar historial de chat (usamos el API de chat nativo de Streamlit)

# Muestra el historial desde la memoria de Streamlit
for message in st.session_state.chat_history.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Input de la pregunta del usuario
user_question = st.chat_input("Pregunta algo sobre el archivo IVA.pdf...")

if user_question and st.session_state.is_setup_done:
    
    # Mostrar la pregunta del usuario en el chat
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generar y mostrar una respuesta
    with st.spinner("TITI está pensando..."):
        ai_response_content = handle_user_input(user_question)

    if ai_response_content:
        with st.chat_message("assistant"):
            st.markdown(ai_response_content)

elif user_question and not st.session_state.is_setup_done:
    st.warning("El Ayudante Impositivo no está listo. Verifica la clave API.")