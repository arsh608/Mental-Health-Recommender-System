"""
Flask web app for MindPath UI
Serves the modern HTML interface and provides API endpoints for mood detection and recommendations.
"""

import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommender.pipeline import load_all, recommend, get_all_content
from models.nlp_mood_detector import get_detector, MOOD_DISPLAY, MOOD_EMOJI, MOOD_COLOR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Start loading models in the background so the server boots instantly
import threading
logger.info("Starting background thread to load models...")
threading.Thread(target=load_all, daemon=True).start()
logger.info("Server starting up...")


@app.route('/')
def index():
    """Serve the main HTML UI"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze mood and get recommendations.
    Returns full mood result + ranked recommendations.
    """
    try:
        data = request.get_json()
        user_text = data.get('text', '').strip()
        stress_level = data.get('stress_level', 5)
        top_n = data.get('top_n', 8)

        if not user_text:
            return jsonify({"error": "No text provided"}), 400

        logger.info(f"Analyzing text: {user_text[:100]}...")

        # Get mood detection and recommendations
        mood_result, recommendations_df = recommend(
            user_text=user_text,
            stress_level=stress_level,
            top_n=top_n
        )

        # Format recommendations
        recommendations = []
        if recommendations_df is not None and len(recommendations_df) > 0:
            for _, row in recommendations_df.iterrows():
                recommendations.append({
                    'title': str(row.get('title', 'Untitled')),
                    'description': str(row.get('description', ''))[:300],
                    'content_type': str(row.get('type', 'Article')),
                    'tfidf_score': float(row.get('cosine_score', 0.5)),
                    'final_score': float(row.get('final_score', 0.5)),
                    'link': str(row.get('link', '#')),
                    'tags': str(row.get('tags', '')),
                    'difficulty': str(row.get('difficulty', 'beginner')),
                    'duration': str(row.get('duration', '—')),
                    'mood_tags': str(row.get('mood_tags', '')),
                })

        response = {
            'mood_result': {
                'primary_mood': mood_result.get('primary_mood', 'unknown'),
                'display_mood': mood_result.get('display_name', 'Unknown'),
                'confidence': float(mood_result.get('confidence', 0.5)),
                'emoji': mood_result.get('emoji', '🌿'),
                'color': mood_result.get('color', '#6b7280'),
                'all_scores': mood_result.get('all_scores', {}),
                'layer_scores': mood_result.get('layer_scores', {}),
                'top_moods': mood_result.get('top_moods', []),
                'sentiment': mood_result.get('sentiment', {'polarity': 0, 'subjectivity': 0}),
            },
            'recommendations': recommendations
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/browse', methods=['GET'])
def browse():
    """
    Return full content library with optional filters.
    Query params: type, difficulty, keyword
    """
    try:
        all_content = get_all_content()

        type_filter = request.args.get('type', '').strip().lower()
        diff_filter = request.args.get('difficulty', '').strip().lower()
        keyword = request.args.get('keyword', '').strip().lower()

        df = all_content.copy()

        if type_filter and type_filter != 'all':
            df = df[df['type'].str.lower() == type_filter]
        if diff_filter and diff_filter != 'all':
            df = df[df['difficulty'].str.lower() == diff_filter]
        if keyword:
            mask = (
                df['title'].str.lower().str.contains(keyword, na=False) |
                df['tags'].str.lower().str.contains(keyword, na=False) |
                df['description'].str.lower().str.contains(keyword, na=False)
            )
            df = df[mask]

        items = []
        for _, row in df.iterrows():
            items.append({
                'title': str(row.get('title', '')),
                'description': str(row.get('description', '')),
                'content_type': str(row.get('type', 'article')),
                'difficulty': str(row.get('difficulty', 'beginner')),
                'duration': str(row.get('duration', '—')),
                'tags': str(row.get('tags', '')),
                'mood_tags': str(row.get('mood_tags', '')),
                'link': str(row.get('link', '#')),
            })

        return jsonify({'items': items, 'count': len(items)})

    except Exception as e:
        logger.error(f"Error in browse endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/nlp-explore', methods=['POST'])
def nlp_explore():
    """
    Run NLP mood detection on text and return full per-layer breakdown.
    """
    try:
        data = request.get_json()
        user_text = data.get('text', '').strip()

        if not user_text:
            return jsonify({"error": "No text provided"}), 400

        detector = get_detector()
        if not detector._ready:
            detector.load()

        result = detector.detect(user_text)

        response = {
            'primary_mood': result.get('primary_mood', 'unknown'),
            'display_name': result.get('display_name', 'Unknown'),
            'confidence': float(result.get('confidence', 0.5)),
            'emoji': result.get('emoji', '🌿'),
            'color': result.get('color', '#6b7280'),
            'all_scores': result.get('all_scores', {}),
            'layer_scores': result.get('layer_scores', {}),
            'top_moods': result.get('top_moods', []),
            'sentiment': result.get('sentiment', {'polarity': 0, 'subjectivity': 0}),
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in nlp-explore endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
