from flask import Flask, render_template, request, jsonify, session
from app.components.llm import llm_chain
from app.common.logger import logger
from app.common.exption import CustomException
from dotenv import load_dotenv
import os
from datetime import timedelta

load_dotenv()
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

app = Flask(__name__, template_folder='templates')

# Session configuration
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
app.config['SESSION_REFRESH_EACH_REQUEST'] = True

from markupsafe import Markup

def nltobr(text):
    """Convert newline characters to HTML line breaks"""
    return Markup(text.replace('\n', '<br>'))

app.jinja_env.filters['nltobr'] = nltobr


@app.route('/', methods=['GET'])
def index():
    """Render the main chat interface"""
    try:
        session.permanent = True
        if 'chat_history' not in session:
            session['chat_history'] = []
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error loading index page: {str(e)}")
        return render_template('index.html'), 200


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and return responses from the LLM"""
    try:
        data = request.get_json()
        user_query = data.get('message', '').strip()
        
        if not user_query:
            return jsonify({
                'error': 'Please enter a message',
                'success': False
            }), 400
        
        if len(user_query) > 5000:
            return jsonify({
                'error': 'Message is too long. Maximum 5000 characters allowed.',
                'success': False
            }), 400
        
        # Add user message to chat history
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append({
            'role': 'user',
            'content': user_query
        })
        
        # Get response from LLM
        logger.info(f"Processing user query: {user_query[:100]}")
        response = llm_chain(user_query)
        
        # Add assistant response to chat history
        session['chat_history'].append({
            'role': 'assistant',
            'content': response
        })
        
        session.modified = True
        
        logger.info("Chat response sent successfully")
        return jsonify({
            'success': True,
            'response': response,
            'message': user_query
        }), 200
        
    except CustomException as ce:
        logger.error(f"Custom exception in chat: {str(ce)}")
        error_msg = "Error processing your request. The knowledge base may not have information about this topic. Please try another question or consult a healthcare professional."
        return jsonify({
            'error': error_msg,
            'success': False
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error in chat: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred. Please try again later.',
            'success': False
        }), 500


@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear the chat history"""
    try:
        session['chat_history'] = []
        session.modified = True
        return jsonify({
            'success': True,
            'message': 'Chat history cleared'
        }), 200
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to clear history'
        }), 500


@app.route('/get-history', methods=['GET'])
def get_history():
    """Get the current chat history"""
    try:
        history = session.get('chat_history', [])
        return jsonify({
            'success': True,
            'history': history
        }), 200
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve history'
        }), 500


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    logger.warning(f"404 Error: {str(error)}")
    return render_template('index.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"500 Error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500


@app.before_request
def before_request():
    """Before each request"""
    session.permanent = True
    app.permanent_session_lifetime = timedelta(hours=1)


if __name__ == '__main__':
    logger.info("Starting Medical RAG Chatbot Flask Application")
    app.run(debug=True, host='0.0.0.0', port=5000)

