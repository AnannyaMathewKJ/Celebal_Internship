import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGSystem:
    def __init__(self, persist_directory=".Week7_Anannya_Mathew/chroma_db", model_name="llama3"):
        print("Initializing Embedding and Language Models...")
        
        # FIXED: Use a dedicated embedding model instead of llama3
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text") 
        
        # Keep llama3 here for generating the final chat answers
        self.llm = ChatOllama(model=model_name, temperature=0.2)
        self.persist_directory = persist_directory
        self.vector_store = None

    def process_pdf(self, pdf_path: str):
        """Loads a PDF, chunks the text, embeds it, and saves it to a local vector store."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Could not find the file: {pdf_path}")
            
        print(f"\n[1/4] Loading PDF: {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        print("[2/4] Splitting text into meaningful chunks...")
        # Chunking: 1000 characters per chunk with a 200 character overlap so context isn't lost mid-sentence
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks.")

        print("[3/4] Creating embeddings and saving to local Vector DB...")
        # Vector Store: Chroma stores these mathematical representations locally on your hard drive
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("Vector Database successfully updated and saved!")

    def ask_question(self, question: str) -> str:
        """Retrieves context from the database and uses the LLM to generate an answer."""
        if not self.vector_store:
            # If database already exists on disk, load it
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                return "Error: Please ingest a PDF document first before asking questions."

        print(f"\nSearching database for context matching: '{question}'...")
        # 2. Setup the Retriever (Pulls top 3 most relevant text chunks)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # 3. Create a strict prompt grounding the AI to the context
        system_prompt = (
            "You are a helpful assistant answering questions based strictly on the provided context.\n"
            "If you do not know the answer or if it's not in the context, say 'I cannot find that in the documents.'\n"
            "Do not make things up.\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 4. Chain everything together: Retrieve -> Augment Prompt -> Generate Answer
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Run the chain
        response = rag_chain.invoke({"input": question})
        return response["answer"]