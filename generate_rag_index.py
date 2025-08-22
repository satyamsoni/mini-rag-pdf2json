import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Load local embedding model, we can change the model depens upon which model is used by LLM
MODEL_NAME = "BAAI/bge-small-en-v1.5"
print(f"🔹 Loading model: {MODEL_NAME} ...")
embed_model = SentenceTransformer(MODEL_NAME, device="cpu")  # change to "cuda" if GPU available

def get_embedding(text):
    # Normalize for cosine similarity retrieval later
    return embed_model.encode([text], normalize_embeddings=True)[0].tolist()

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

def create_rag_index(chunks, output="rag_index.json"):
    rag_index = []
    for chunk in tqdm(chunks, desc="🔹 Creating embeddings"):
        text = chunk.page_content.strip()
        if not text:
            continue
        vector = get_embedding(text)
        rag_index.append({
            "text": text,
            "embedding": vector
        })

    with open(output, "w") as f:
        json.dump(rag_index, f)
    #Color of success
    print(f"Saved to {output}")

if __name__ == "__main__": 
    chunks = process_pdf("myfile.pdf")
    create_rag_index(chunks)
