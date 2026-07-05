import streamlit as st
import os
import shutil
from app import RAGSystem

# 1. Page Configuration Setup
st.set_page_config(page_title="RAG Document Assistant", page_icon="🤖", layout="wide")

# Initialize RAG System in Streamlit Session State so it stays active across clicks
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 2. Sidebar Layout (RAG Controls)
with st.sidebar:
    st.header("⚙️ RAG Controls")
    
    # File Upload Widget
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    # Model Selection (Defaulted to Llama 3)
    model_option = st.selectbox("LLM Model", ["llama3", "mistral", "phi3"], index=0)
    
    # Configuration Sliders for Extra Tuning
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    
    process_button = st.button("🔧 Search & Process Document", use_container_width=True)

    # File Ingestion Logic (Indented inside the sidebar context)
    if process_button and uploaded_file is not None:
        with st.spinner("Processing document... Converting to vectors..."):
            
            # FIXED: Instead of deleting the folder, clear the collection if it exists
            if st.session_state.rag_system.vector_store is not None:
                try:
                    # This safely clears old data without breaking the file lock
                    st.session_state.rag_system.vector_store.delete_collection()
                except Exception:
                    pass # Fail silently if the collection doesn't exist yet
            
            # Save uploaded file temporarily to pass to the RAG pipeline
            temp_file_path = f"./temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Update model settings on the fly if user changed it
                st.session_state.rag_system.llm.model = model_option
                
                # Process the PDF
                st.session_state.rag_system.process_pdf(temp_file_path)
                st.success("🎉 Successfully loaded and indexed!")
                st.session_state.chat_history = [] # Clear old chat history for new doc
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                # Cleanup the temporary file from disk
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
    elif process_button and uploaded_file is None:
        st.warning("Please upload a PDF file first!")

# 3. Main Chat View Panel
st.title("💬 Chat with your Documents")
st.caption("Ask questions grounded strictly to your private data context.")

# Display existing chat log history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input Box
if user_query := st.chat_input("What is the main idea of the research notes?"):
    
    # Display human message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Generate and display AI response using the RAG backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.rag_system.ask_question(user_query)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")