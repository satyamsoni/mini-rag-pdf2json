# RAG Generation Sample implementation 
Learn RAG Creation from PDF to JSON , which can be used to query small RAG Data while creating LLM+RAG

A simple small example i made using Cloudflare Worker AI + R2 ( S3 type storage ) with output of this RAG program to create Chatbot with small RAG data.

## Purpose : 
Generate small RAG data ( basically Vector embeddings) which can be used to query.



## How to run

Please change PDF file name in "generate_rag_index.py"
python3 -m venv venv
source venv/bin/activate
run "python generate_rag_index.py" 
