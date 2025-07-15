**RAG-Based Legal Chatbot with Streaming Interface**
This is a project submission for the Junior AI Engineer assignment from Amlgo Labs.
It demonstrates a working RAG (Retrieval-Augmented Generation) pipeline using an open-source LLM, FAISS-based vector search, and a real-time Streamlit chatbot interface.
#-----------------------------------------------------------------------------------------------------------------------------------
**Project Objective**
Build a chatbot that can answer user questions based on a custom legal document (like Terms & Conditions or Privacy Policy) using:

1-Document chunking + embeddings

2-FAISS for semantic search

3-Hugging Face open-source LLM (Flan-T5)

4-Real-time streamed answers

5-Easy-to-use Streamlit UI
#-----------------------------------------------------------------------------------------------------------------------------------
**How It Works (Simplified Flow):**
[User Question]
[Retrieve relevant chunks from FAISS Vector DB]
[Inject question + context into LLM prompt]
[LLM (Flan-T5) generates grounded answer]
[Streamed response + source shown in UI]

#-----------------------------------------------------------------------------------------------------------------------------------
**Component:	                Library/Tool Used:**

LLM (language model)	    google/flan-t5-base (via Transformers)
Vector DB	                FAISS (local semantic search)
Embeddings	                sentence-transformers
Frontend	                Streamlit (Python UI)
Backend Pipeline	        LangChain (optional modules)


#-----------------------------------------------------------------------------------------------------------------------------------
**Installation Steps:**
1. Clone this repo
git clone https://github.com/<your-username>/rag_chatbot_amlgo.git
cd rag_chatbot_amlgo

2. Create and activate virtual environment (Windows)
python -m venv Amlgo_env
.\Amlgo_env\Scripts\activate

3. Install required libraries
pip install -r requirements.txt

4. Place your document
Replace document.txt with your own Terms & Conditions or Privacy Policy file.

#-----------------------------------------------------------------------------------------------------------------------------------
**Run the Application :**

Step 1: Embed your document
python ingest.py
This will:
Load and clean your document
Break it into chunks (100–300 words)
Generate semantic embedding
Store embeddings in a FAISS vector DB

Step 2: Launch chatbot UI
streamlit run app.py

Now go to http://localhost:8501

#-----------------------------------------------------------------------------------------------------------------------------------
**Notes :**
I used google/flan-t5-base instead of mistralai/Mistral-7B-Instruct-v0.1 due to Hugging Face gated access restrictions.
You can swap models in generator.py anytime.
Add more documents or switch to Qdrant/ChromaDB if needed.


#-----------------------------------------------------------------------------------------------------------------------------------
Yash Jain
Passionate about AI, NLP, and building real-world LLM apps,Pyhton,Numpy,Pandas and Visualization.
Feel free to reach out for collaboration!