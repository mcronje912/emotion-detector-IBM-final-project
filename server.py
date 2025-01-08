"""
This module implements a web server for emotion detection in text.
It provides endpoints for serving the web interface and processing
emotion detection requests using a pre-trained NLP model.
"""

from flask import Flask, render_template, request # type: ignore
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")

@app.route("/")
def render_index_page():
    """Render the main application page.
    Returns:
        str: Rendered HTML content of the index page
    """
    return render_template('index.html')

@app.route("/emotionDetector")
def emotion_detector_route():
    """Process emotion detection requests and return formatted results.
    Returns:
        str: Formatted string containing emotion analysis or error message
    """
    text_to_analyze = request.args.get('textToAnalyze')
    response = emotion_detector(text_to_analyze)
    if response['dominant_emotion'] is None:
        return "Invalid text! Please try again!"
    formatted_response = (
        f"For the given statement, the system response is "
        f"'anger': {response['anger']}, "
        f"'disgust': {response['disgust']}, "
        f"'fear': {response['fear']}, "
        f"'joy': {response['joy']} and "
        f"'sadness': {response['sadness']}. "
        f"The dominant emotion is {response['dominant_emotion']}."
    )
    return formatted_response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
    