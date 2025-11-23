RAG Enhanced Agent:
An intelligent conversational AI assistant powered by Claude (Anthropic) with Retrieval-Augmented Generation (RAG) capabilities, built with Streamlit and LangChain.
Features

1. Document Processing: Upload and index PDF, DOCX, and TXT files for context-aware conversations
2. RAG Integration: Uses FAISS vector store with HuggingFace embeddings for semantic document retrieval
3. Streaming Responses: Real-time Claude AI responses with smooth typing animation
4. Modern UI: Dark mode interface with chat bubble design for enhanced user experience
5. Customizable System Prompts: Configure the AI's behavior and personality
6. Web Search Ready: Toggle for web search integration (extensible)
7. Chat History: Persistent conversation memory within session

Tech Stack

AI Model: Claude Sonnet 4.5 (Anthropic)
Framework: Streamlit
Vector Database: FAISS
Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
Document Processing: LangChain Community
Environment Management: python-dotenv

Installation

Clone the repository
```
bashgit clone <your-repo-url>
cd rag-enhanced-agent

Install dependencies

```
bashpip install streamlit anthropic python-dotenv langchain-community faiss-cpu sentence-transformers pypdf docx2txt
```

3. **Set up environment variables**
```
Create a `.env` file in the root directory:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
Configure document path
Update DOCS_PATH in the script to point to your documents folder:
pythonDOCS_PATH = "/path/to/your/documents"
Usage

Run the application
```
bash streamlit run app.py


2. **Select documents** from the sidebar to enable RAG
3. **Toggle web search** if needed (placeholder for future implementation)
4. **Customize system prompt** to adjust AI behavior
5. **Start chatting** with your AI assistant!

## Document Support

- **PDF**: `.pdf` files
- **Word**: `.docx` files
- **Text**: `.txt` files

Documents are automatically chunked (1000 chars with 150 overlap) and embedded for semantic search.

## Configuration

### Model Parameters
- **Model**: `claude-sonnet-4-5`
- **Max Tokens**: 1500
- **Temperature**: 0.8
- **Retrieval**: Top 3 similar documents

### Customization
- Adjust `chunk_size` and `chunk_overlap` in `create_vector_store()`
- Modify `k` parameter in `retrieve_relevant_docs()` for more/fewer retrieved documents
- Change streaming delay in `generate_streaming_response()` for faster/slower typing effect

## Privacy Features

Built-in content filtering for sensitive information (configurable in system prompt).
```
## Requirements.txt

streamlit  
anthropic  
python-dotenv  
langchain-community  
faiss-cpu  
sentence-transformers  
pypdf  
docx2txt  
Future Enhancements

1.Actual web search integration  
2.Multi-file upload support  
3.Export chat history  
4.Custom embedding model selection  
5.Advanced document filtering  
6.Multi-language support
