"""
MindPath - Mental Health Content Recommender System
Entry point for Flask web application
"""

import sys
from pathlib import Path

# Add the project root to the path so imports work correctly
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

# Run the Flask app
from app.flask_app import app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MindPath - Mental Health Recommender System")
    print("="*60)
    print("Starting Flask server...")
    print("Open browser: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)
