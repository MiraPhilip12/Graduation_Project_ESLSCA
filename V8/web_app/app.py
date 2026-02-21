from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.performance_analyzer import PerformanceAnalyzer
from utils.visualization import VisualizationUtils

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Available actions
AVAILABLE_ACTIONS = [
    "Paddle_forehand", "Forehand_lob", "Backhand", 
    "Backhand_lob", "Smash", "Phone_call", 
    "Checking_watch", "Clapping", "Hand_shake", "Thumbs_up"
]

# Age groups
AGE_GROUPS = ["teenager", "young_adult", "adult", "senior"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', actions=AVAILABLE_ACTIONS, age_groups=AGE_GROUPS)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video upload and analysis"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Get form data
    actor_name = request.form.get('actor_name', 'Unknown')
    actor_age = request.form.get('actor_age', '')
    selected_actions = request.form.getlist('actions')
    age_group = request.form.get('age_group', 'teenager')
    
    if not selected_actions:
        return jsonify({'error': 'Please select at least one action to assess'}), 400
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{actor_name.replace(' ', '_')}_{timestamp}.mp4"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Start analysis (we'll do this in background in production)
    # For now, analyze the first selected action
    target_action = selected_actions[0]
    
    try:
        # Initialize analyzer
        analyzer = PerformanceAnalyzer()
        
        # Run analysis
        results = analyzer.analyze_video(filepath, target_action)
        
        # Add actor info
        results['actor_name'] = actor_name
        results['actor_age'] = actor_age
        results['age_group'] = age_group
        results['selected_actions'] = selected_actions
        results['video_filename'] = filename
        
        # Generate summary
        summary = analyzer.generate_summary(results)
        
        # Save results
        results_path = os.path.join(app.config['REPORTS_FOLDER'], f"results_{timestamp}.json")
        analyzer.save_results(results, results_path)
        
        # Generate visualizations
        viz = VisualizationUtils()
        
        # Create radar chart
        metrics = results['overall_scores']
        radar_fig = viz.create_radar_chart(metrics, "Performance Metrics")
        radar_html = radar_fig.to_html() if hasattr(radar_fig, 'to_html') else ""
        
        # Prepare response
        response = {
            'success': True,
            'actor_name': actor_name,
            'target_action': target_action,
            'performance_level': results['performance_level'],
            'final_score': float(results['overall_scores']['final_score']),
            'summary': summary,
            'results_path': results_path,
            'radar_chart': radar_html,
            'timestamp': timestamp
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/report/<timestamp>')
def view_report(timestamp):
    """View analysis report"""
    results_path = os.path.join(app.config['REPORTS_FOLDER'], f"results_{timestamp}.json")
    
    if not os.path.exists(results_path):
        return "Report not found", 404
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return render_template('report.html', results=results, timestamp=timestamp)

@app.route('/api/results/<timestamp>')
def get_results(timestamp):
    """Get analysis results as JSON"""
    results_path = os.path.join(app.config['REPORTS_FOLDER'], f"results_{timestamp}.json")
    
    if not os.path.exists(results_path):
        return jsonify({'error': 'Report not found'}), 404
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return jsonify(results)

@app.route('/api/compare', methods=['POST'])
def compare_actors():
    """Compare multiple actors' performances"""
    data = request.json
    actor_results = data.get('results', [])
    
    if not actor_results:
        return jsonify({'error': 'No results to compare'}), 400
    
    # Create comparison visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    actors = [r.get('actor_name', f"Actor {i+1}") for i, r in enumerate(actor_results)]
    scores = [r.get('final_score', 0) for r in actor_results]
    
    bars = ax.bar(actors, scores, color=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Final Score')
    ax.set_title('Actor Performance Comparison')
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.2f}', ha='center', va='bottom')
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return jsonify({
        'comparison_chart': img_str,
        'actors': actors,
        'scores': scores
    })

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download a file"""
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)