import os
import glob
from dotenv import load_dotenv

# Document parsing and splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store and Local Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Google LLM Integration Layer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def main():

    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "❌ GOOGLE_API_KEY is missing from your .env file."
        )

    pdf_files = glob.glob("data/*.pdf")

    if not pdf_files:
        print("❌ No PDF found inside data folder.")
        return

    pdf_path = pdf_files[0]

    # ---------------------------------------------------
    # EMBEDDING MODEL
    # ---------------------------------------------------
    print("🧠 Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    INDEX_PATH = "faiss_index"

    # ---------------------------------------------------
    # LOAD OR CREATE VECTOR STORE
    # ---------------------------------------------------
    if os.path.exists(INDEX_PATH):

        print("✅ Existing FAISS index found.")
        print("📦 Loading vector database from disk...")

        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        print(f"📄 Loading document: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        print("✂️ Splitting document into chunks...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )

        docs = text_splitter.split_documents(documents)

        print(f"📑 Created {len(docs)} chunks")

        print("📦 Creating FAISS index...")

        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        print("💾 Saving FAISS index to disk...")

        vectorstore.save_local(INDEX_PATH)

        print("✅ Index saved successfully")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # ---------------------------------------------------
    # LLM
    # ---------------------------------------------------
    print("🤖 Connecting to Gemini...")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    # ---------------------------------------------------
    # PROMPT
    # ---------------------------------------------------
    system_prompt = (
        "You are a helpful reading assistant.\n"
        "Answer the question using ONLY the retrieved context.\n"
        "If the answer is not present, say so.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    print("\n✅ PDF RAG Chatbot Ready!")
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:

        question = input("\n👤 You: ")

        if question.lower().strip() == "exit":
            print("👋 Goodbye!")
            break

        if not question.strip():
            continue

        print("🤖 Searching knowledge base...")

        try:

            response = rag_chain.invoke(
                {"input": question}
            )

            print(f"\n🤖 Bot: {response['answer']}")

            print("\n📌 Context Sources:")

            for i, doc in enumerate(
                response.get("context", [])
            ):

                page = doc.metadata.get(
                    "page",
                    "N/A"
                )

                snippet = (
                    doc.page_content
                    .replace("\n", " ")[:90]
                )

                print(
                    f"[{i+1}] Page {page}: "
                    f"\"{snippet}...\""
                )

            print("-" * 50)

        except Exception as e:

            print(f"\n❌ Error: {e}")
            print("-" * 50)


if __name__ == "__main__":

    os.makedirs("data", exist_ok=True)

    main()