# 🌐 Website Q&A Bot

An interactive **Streamlit app** that allows you to ask questions about the content of any website.  
It uses **LangChain**, **Hugging Face models**, and **FAISS embeddings** to retrieve and answer questions from web pages.  

## 🚀 Features
- Load any website URL and extract text
- Store content in a vector database (FAISS)
- Query the website using natural language
- Powered by Hugging Face LLM + LangChain
- Clean Streamlit UI for interaction

## ⚡ Tech Stack
- Python
- LangChain
- Hugging Face (LLM + Embeddings)
- FAISS (Vector Store)
- Streamlit (UI)

## ▶️ Run Locally
```bash
git clone https://github.com/your-username/website-qa-bot.git
cd website-qa-bot
pip install -r requirements.txt
streamlit run app.py
