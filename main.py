import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory

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
        # Mostrar mensaje de advertencia si no se ingresa la clave
        if not input_text:
            st.warning("⚠️ Por favor, ingresa tu clave API para comenzar.")
        
    return input_text

openai_api_key = get_openai_api_key()

# Ruta estática al archivo PDF
PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "IVA.pdf")
# La base de datos vectorial se guarda en el mismo directorio que el script
VECTOR_DB_PATH = "faiss_index_iva" 

# Función para cargar y procesar el documento
@st.cache_resource
def process_document(api_key: str):
    """
    Carga el PDF, aplica splits, embeddings y crea el Vector Store.
    Esto solo se ejecuta una vez gracias a st.cache_resource.
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
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Crear la base de datos vectorial (Vector Store)
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el documento o crear embeddings: {e}")
        st.stop()


# Función para inicializar la cadena RAG con memoria
def get_conversation_chain(vectorstore):
    """
    Crea y retorna la cadena de conversación con memoria.
    """
    # Inicializar el modelo de chat con la clave API
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.7, 
        openai_api_key=openai_api_key
    )
    
    # Memoria para el chat
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    # Crear la cadena RAG
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        # Opcional: para que las respuestas sean en español
        chain_type="stuff",
        verbose=True
    )
    return conversation_chain


# --- 3. LÓGICA DE CHAT Y ESTADO ---

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_setup_done" not in st.session_state:
    st.session_state.is_setup_done = False


def setup_backend():
    """Función que inicia el backend (solo se llama una vez)."""
    if openai_api_key:
        with st.spinner("Procesando documentos y preparando el Ayudante Impositivo..."):
            # 1. Procesar el documento (obtiene el vector store)
            vectorstore = process_document(openai_api_key)
            
            # 2. Inicializar la cadena de conversación
            st.session_state.conversation = get_conversation_chain(vectorstore)
            
            # Marcar como listo
            st.session_state.is_setup_done = True
        st.success("¡TITI-AYUDANTE IMPOSITIVO listo! Ya puedes preguntar.")

if not st.session_state.is_setup_done and openai_api_key:
    setup_backend()
elif not openai_api_key:
    # Mostrar mensaje si falta la clave, ya se maneja en get_openai_api_key, pero reforzamos
    st.info("Ingresa tu clave API en la barra lateral para cargar los datos del IVA.")


def handle_user_input(user_question):
    """Maneja la pregunta del usuario, llama al RAG y actualiza el historial."""
    if st.session_state.conversation is None:
        st.error("El modelo aún no está configurado. Por favor, verifica tu clave API y espera la carga inicial.")
        return

    # Llamar a la cadena de conversación RAG
    response = st.session_state.conversation({'question': user_question})
    
    # La memoria ya actualizó el historial internamente. Aquí lo extraemos y mostramos.
    st.session_state.chat_history = response['chat_history']


# --- 4. INTERFAZ DE CONVERSACIÓN ---

# Input de la pregunta del usuario
user_question = st.chat_input("Pregunta algo sobre el archivo IVA.pdf...")

if user_question and st.session_state.is_setup_done:
    # Añadir la pregunta del usuario al historial
    st.session_state.chat_history.append(("user", user_question))
    
    # Generar y mostrar una respuesta (usamos una barra de progreso mientras responde)
    with st.spinner("TITI está pensando..."):
        handle_user_input(user_question)

elif user_question and not st.session_state.is_setup_done:
    st.warning("El Ayudante Impositivo no está listo. Verifica que la clave API esté ingresada y que el PDF se haya cargado correctamente.")


# Mostrar historial de chat
st.markdown("### 💬 Historial de Conversación")

if st.session_state.chat_history:
    # Mostrar el historial en orden inverso para que la última conversación esté abajo
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        role, content = (message.type, message.content) if hasattr(message, 'type') else ('user' if i % 2 == 0 else 'assistant', message)
        
        # Usar los elementos nativos de chat de Streamlit
        if role == 'user':
            with st.chat_message("user"):
                st.write(content)
        else: # assistant
            with st.chat_message("assistant"):
                st.write(content)