# import streamlit as st
# from langchain.vectorstores import Qdrant
# from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceEmbeddings
# from qdrant_client import QdrantClient
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.vectorstores import Qdrant
# import os

# from dotenv import load_dotenv
# load_dotenv()


# # Load environment variables
# QDRANT_URL = os.environ["QDRANT_URL"]
# QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
# GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# # Embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Connect to Qdrant
# client = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY
# )

# qdrant = Qdrant(
#     client=client,
#     collection_name="otic-knowledge-base",
#     embeddings=embeddings
# )


# retriever = qdrant.as_retriever()

# # Gemini LLM setup
# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-1.5-flash-latest",  # You can also try "gemini-pro" if available
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0.2
# )

# # QA Chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=False
# )

# # Streamlit UI
# st.set_page_config(page_title="OTIC Chatbot", page_icon="🤖")
# st.title("🤖 OTIC Interaction Bot")
# st.write("Ask me anything about OTIC Foundation.")

# query = st.text_input("You:", placeholder="e.g. What is the mission of OTIC Foundation?")

# if query:
#     with st.spinner("Thinking..."):
#         try:
#             answer = qa_chain.run(query)
#             st.markdown(f"**You asked:** {query}")
#             st.markdown(f"**OTIC Bot:** {answer}")
#         except Exception as e:
#             st.error(f"⚠️ Error: {e}")




import streamlit as st
import os
from langchain.chains import RetrievalQA
from collections import Counter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import datetime

from dotenv import load_dotenv
load_dotenv()

# Load environment variables
if st.secrets.get("QDRANT_URL"):
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    load_dotenv()
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# QDRANT_URL = os.environ["QDRANT_URL"]
# QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
# GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# Set your Google API key for Gemini
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Connect to existing Qdrant collection
qdrant = Qdrant(
    client=client,
    collection_name="otic-knowledge-base",
    embeddings=embeddings
)

# Create retriever
retriever = qdrant.as_retriever(search_kwargs={"k": 6})

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # Important for showing source chunks
)

# ----------------- Streamlit UI -------------------
st.set_page_config(page_title="OTIC Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 OTIC Foundation AI Assistant")
st.markdown("Ask anything about **Otic Foundation** and get instant answers!")

# Session state to store Q&A history and FAQ tracking
if "history" not in st.session_state:
    st.session_state.history = []  # stores (question, answer, timestamp)
if "faq_counter" not in st.session_state:
    st.session_state.faq_counter = Counter()

# Input box
query = st.text_input("💬 Type your question here:")

if query:
    with st.spinner("Thinking..."):
        try:
            response = qa_chain.invoke({"query": query})
            answer = response["result"]

            # Save to history
            st.session_state.history.append({
                "question": query,
                "answer": answer,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Count FAQ occurrences
            st.session_state.faq_counter[query.lower()] += 1

            st.success("Done!")
            st.markdown(f"** You asked:** {query}")
            st.markdown(f"** OTIC Bot:** {answer}")

        except Exception as e:
            st.error(f"Error: {e}")

# Display conversation history
with st.expander("📜 Conversation History"):
    for i, item in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**Q{i+1}:** {item['question']}")
        st.markdown(f"**A{i+1}:** {item['answer']}")
        st.caption(f"🕒 {item['time']}")
        st.write("---")

# Display top FAQs
with st.expander("🔥 Frequently Asked Questions"):
    if st.session_state.faq_counter:
        for question, count in st.session_state.faq_counter.most_common(5):
            st.markdown(f"**{question.capitalize()}** — Asked {count} times")
    else:
        st.info("No FAQs yet. Start asking questions!")




# <div style="text-align:center;">
#   <iframe 
#       src="https://oticaiagent-p47svvzvua5hdnjxvxoywh.streamlit.app/" 
#       width="100%" 
#       height="600" 
#       style="border:1px solid #ddd; border-radius:10px;">
#   </iframe>
# </div>
