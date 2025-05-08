from flask import Flask, request, jsonify, render_template_string
import os
from rag_assistant import load_documents, process_documents, VectorStore, process_query
from waitress import serve

# Initialize the application
app = Flask(__name__)

# Load documents and initialize vector store
print("Loading documents...")
documents = load_documents()
if not documents:
    print("No documents found. Please add documents to the 'docs' directory.")
else:
    print(f"Loaded {len(documents)} documents.")

processed_chunks = process_documents(documents)
vector_store = VectorStore()
vector_store.add_documents(processed_chunks)
print("Vector store initialized.")

# HTML template for the frontend
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Assistant</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .chat-container {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 20px;
            margin-top: 20px;
            height: 400px;
            overflow-y: auto;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e6f7ff;
            margin-left: 20%;
            margin-right: 5px;
        }
        .assistant-message {
            background-color: #f2f2f2;
            margin-right: 20%;
            margin-left: 5px;
        }
        .input-area {
            display: flex;
            margin-top: 20px;
        }
        #query-input {
            flex-grow: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            margin-left: 10px;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .source {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>RAG Assistant</h1>
    <div class="chat-container" id="chat-container">
        <div class="message assistant-message">
            Hello! I'm your RAG Assistant. I can answer questions based on my knowledge or use special tools for calculations and definitions. How can I help you today?
        </div>
    </div>
    <div class="input-area">
        <input type="text" id="query-input" placeholder="Ask a question...">
        <button onclick="sendQuery()">Send</button>
    </div>

    <script>
        function sendQuery() {
            const queryInput = document.getElementById('query-input');
            const query = queryInput.value.trim();
            
            if (query === '') return;
            
            // Add user message to chat
            addMessage(query, 'user');
            
            // Clear input
            queryInput.value = '';
            
            // Send query to server
            fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            })
            .then(response => response.json())
            .then(data => {
                // Add assistant response to chat
                addMessage(data.response, 'assistant', data.source);
            })
            .catch(error => {
                console.error('Error:', error);
                addMessage('Sorry, there was an error processing your request.', 'assistant');
            });
        }
        
        function addMessage(text, sender, source = null) {
            const chatContainer = document.getElementById('chat-container');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            
            messageDiv.textContent = text;
            
            if (source) {
                const sourceSpan = document.createElement('div');
                sourceSpan.className = 'source';
                sourceSpan.textContent = `Source: ${source}`;
                messageDiv.appendChild(sourceSpan);
            }
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Allow Enter key to send message
        document.getElementById('query-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendQuery();
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Render the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/query', methods=['POST'])
def query():
    """Process query from the frontend"""
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({'response': 'Please provide a query.', 'source': None})
    
    # Process the query
    result = process_query(query_text, vector_store)
    
    # Extract source information if available
    source = None
    if result.get('context') and result['context']:
        sources = [ctx.get('source') for ctx in result['context'] if ctx.get('source')]
        if sources:
            source = sources[0]
    
    return jsonify({
        'response': result['answer'],
        'source': source
    })

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Check if running in production or development
    if os.environ.get('FLASK_ENV') == 'production':
        print(f"Starting production server on port {port}...")
        serve(app, host='0.0.0.0', port=port)
    else:
        print(f"Starting development server on port {port}...")
        # Development mode
        app.run(host='0.0.0.0', port=port, debug=True) 