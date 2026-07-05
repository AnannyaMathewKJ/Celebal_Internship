import os
from app import RAGSystem

def main():
    print("=" * 50)
    print("      Welcome to your Private Document RAG System      ")
    print("=" * 50)
    
    # Prerequisite: Ensure the user has Ollama running locally
    print("💡 Make sure Ollama is installed and running (`ollama run llama3`)")
    
    # Initialize the system
    rag = RAGSystem()

    # Step A: Document Ingestion check
    if not os.path.exists("./chroma_db"):
        print("\nNo active vector database found. Let's process a document first.")
        pdf_path = input("Enter the absolute or relative path to your PDF file: ").strip()
        try:
            rag.process_pdf(pdf_path)
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return
    else:
        print("\nFound an existing Vector Database! Ready to chat.")
        change_doc = input("Do you want to upload a NEW PDF instead? (y/n): ").strip().lower()
        if change_doc == 'y':
            pdf_path = input("Enter the path to the new PDF file: ").strip()
            try:
                # Remove old DB index to start fresh
                import shutil
                shutil.rmtree("./chroma_db")
                rag.process_pdf(pdf_path)
            except Exception as e:
                print(f"❌ Error processing file: {e}")
                return

    # Step B: The Interactive QA Loop
    print("\nInitialization Complete. You can now ask questions about your document!")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        user_query = input("\n💬 Ask a question about your document: ").strip()
        
        if user_query.lower() in ['exit', 'quit']:
            print("Shutting down the system. Goodbye!")
            break
            
        if not user_query:
            continue
            
        try:
            answer = rag.ask_question(user_query)
            print(f"\n🤖 AI Answer:\n{answer}")
            print("-" * 40)
        except Exception as e:
            print(f"❌ An error occurred while generating the answer: {e}")

if __name__ == "__main__":
    main()