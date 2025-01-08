from transformers import pipeline # type: ignore

def emotion_detector(text_to_analyze):
    """
    Detect emotions in the given text using a pre-trained transformer model.
    
    Args:
        text_to_analyze (str): The text to analyze for emotions
        
    Returns:
        dict: Dictionary containing emotion scores and dominant emotion.
             Returns None values for all fields if input is invalid.
    """
    # Check for invalid or empty input
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }
    
    try:
        # Initialize the emotion classifier
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        
        # Get predictions
        result = classifier(text_to_analyze)
        emotions = result[0]
        
        # Create our emotion scores dictionary with default values
        emotion_scores = {
            'anger': 0.0,
            'disgust': 0.0,
            'fear': 0.0,
            'joy': 0.0,
            'sadness': 0.0
        }
        
        # Fill in the actual scores and track the highest score
        max_score = 0.0
        dominant_emotion = None
        
        for item in emotions:
            emotion = item['label'].lower()
            score = float(item['score'])
            
            if emotion in emotion_scores:
                emotion_scores[emotion] = score
                if score > max_score:
                    max_score = score
                    dominant_emotion = emotion
        
        # Add the dominant emotion to our results
        emotion_scores['dominant_emotion'] = dominant_emotion
        
        return emotion_scores

    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }