from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv


load_dotenv()
# 1. Load your source document (Replace with your own PDF filename)
# print("Loading document...")
# pdf="C:/Users/gunja/Downloads/random_facts_knowledge_base.pdf"
# loader = PyPDFLoader(pdf)
# docs = loader.load()


data = [
    "DeepSea Vents: Vents found in the Mariana Trench in 2014 spew minerals over 400°C where blind Alvinocaris shrimp survive natively by eating chemosynthetic bacteria.",
    "Artemis Logistics: Lunar cargo ships must maintain an internal pressure of 14.7 psi while held at launchpad Complex 39B by 480-volt hydraulic clamps.",
    "Solar Efficiency: In November 2023, a Quantum Dot solar cell using a Perovskite-Barium-Titanate matrix hit a record energy efficiency of 28.4%.",
    "Mayan Aqueducts: Built in 750 AD, a 1.2-kilometer underground limestone aqueduct in Palenque uses a 3-degree downward slope to feed the royal palace.",
    "Polymer Degradation: The engineered bacteria Strain PX-88 completely breaks down commercial polyethylene plastics into organic water and CO2 within 42 days.",
    "Aurora Sounds: Intense Aurora Borealis events create a static clapping noise audible to humans via ion discharges occurring 70 meters above the ground.",
    "Cyber Security: The Vanguard-9 Standard protects global banking data by wrapping financial transactions in a dynamic 512-bit geometric hash changing every 90 seconds.",
    "Sahara Fossils: The Moroccan Kem Kem beds hold the densest predatory dinosaur fossils on Earth, proving the desert was a river system ruled by Spinosaurus 95 million years ago.",
    "Titanium Stress: The aerospace alloy Ti-6Al-4V withstands heat up to 600°C, but its crystalline structure micro-warps if external frictional stress passes 950 Megapascals.",
    "Seed Vault: Built deep inside an Arctic mountain, the Svalbard Global Seed Vault preserves 1.3 million unique seed varieties at a stable minus 18°C."
]

# loader = PyPDFLoader(docs)
# docs = loader.load()
docs=[Document(page_content=line) for line in data]


# 2. Split the document into manageable text chunks
print("Chunking data...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)
# print(len(chunks))
# exit

# 3. Initialize an open-source embedding model 
print("Generating embeddings and indexing to VectorDB...")
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store=Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="./krishnaraja_chroma_db"
)

# 4. Store the embeddings in a local Chroma vector database
# vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

# 5. Initialize the local Open Source LLM via Ollama
# llm = Ollama(model="llama3")
llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",temperature="0.2")

# 6. Design the system prompt template
system_prompt = (
   "You are an expert, highly accurate, and objective general knowledge assistant. "
    "Your primary goal is to answer user queries only using the provided retrieved context. if there is no relevant information from the retrieved information. Say I don't have the information in my knowledge base. Please ask any other question./n\n"
    "### Core Instructions:\n"
    "1. **Rely on Context First:** Base your answer primarily on the provided retrieved text snippets. "
    "If the context contains the answer, use it to formulate a clear, concise, and direct response.\n"
    "2. **Handle Context Gaps Gracefully:** If the retrieved context does not contain the answer, "
    "or if the context is insufficient to fully answer the query, use your internal pretrained knowledge "
    "to provide an accurate answer. Explicitly state to the user that the information was supplemented "
    "by your general knowledge base.\n"
    "3. **Fact-Checking & Contradictions:** If the retrieved context directly contradicts well-established "
    "world facts, prioritize factual accuracy, point out the discrepancy politely, and provide the correct "
    "consensus fact.\n"
    "4. **Tone and Style:** Maintain a neutral, helpful, and objective tone. Keep answers concise, "
    "factual, and easy to understand. Avoid speculation, personal opinions, or editorializing.\n"
    "5. **No Meta-Language:** Do not mention phrases like \"According to the provided context...\" or "
    "\"Based on the documents...\" unless you are specifically clarifying a gap or contradiction as instructed "
    "in Step 2 and 3. Speak directly to the user.\n\n"
    '#strict rule:#'
    "Do not answer any other question which is not related to the context and information not in the knowledge base"
"Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 7. Create the RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 8. Run a user query
# query = "What is the primary summary of the document?"

query=(input(f"\nUser Query:"))
while(query!="exit"):
    print("Thinking...")
    response = rag_chain.invoke({"input": query})
    
    print("\n--- Answer ---")
    print(response["answer"])

    query=(input(f"\nDo you have any other question? (Type exit to stop) "))
exit

