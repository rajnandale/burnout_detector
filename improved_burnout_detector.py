import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

class ImprovedBurnoutDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.is_trained = False
        
    def create_synthetic_dataset(self, num_samples=1000):
        """
        Create a balanced synthetic dataset with realistic workplace feedback
        """
        np.random.seed(42)
        
        # Define realistic workplace scenarios
        low_risk_texts = [
            "I enjoy my work and feel productive most days.",
            "The team is supportive and I feel valued.",
            "Work is challenging but manageable.",
            "I have a good work-life balance.",
            "My manager is approachable and helpful.",
            "I feel motivated to come to work.",
            "The workload is reasonable this week.",
            "I'm learning new skills and growing professionally.",
            "My colleagues are collaborative and friendly.",
            "I feel accomplished after completing tasks."
        ]
        
        moderate_risk_texts = [
            "Work has been busy lately, feeling a bit overwhelmed.",
            "The workload is increasing but I'm managing.",
            "Sometimes I feel tired after long meetings.",
            "Deadlines are tight but achievable.",
            "I wish I had more time for breaks.",
            "The project is challenging but interesting.",
            "I'm working extra hours occasionally.",
            "Some days are more stressful than others.",
            "I need to prioritize my tasks better.",
            "The team is under pressure but coping."
        ]
        
        high_risk_texts = [
            "I'm completely exhausted and burnt out from work.",
            "I can't keep up with the workload anymore.",
            "I feel overwhelmed and stressed every day.",
            "I'm working 12-hour days and still behind.",
            "I'm emotionally drained and can't focus.",
            "The stress is affecting my health and sleep.",
            "I feel like I'm drowning in responsibilities.",
            "I'm considering quitting due to burnout.",
            "I'm constantly tired and irritable at work.",
            "The pressure is unbearable and affecting my life."
        ]
        
        # Create balanced dataset
        texts = []
        labels = []
        
        # Low risk (0)
        for _ in range(num_samples // 3):
            base_text = np.random.choice(low_risk_texts)
            # Add some variation
            variations = [
                base_text,
                base_text + " Overall, I'm satisfied with my role.",
                base_text + " The environment is positive.",
                base_text + " I feel supported by management."
            ]
            texts.append(np.random.choice(variations))
            labels.append(0)
        
        # Moderate risk (1)
        for _ in range(num_samples // 3):
            base_text = np.random.choice(moderate_risk_texts)
            variations = [
                base_text,
                base_text + " But I'm managing okay.",
                base_text + " It's temporary, I think.",
                base_text + " I hope it gets better soon."
            ]
            texts.append(np.random.choice(variations))
            labels.append(1)
        
        # High risk (2)
        for _ in range(num_samples // 3):
            base_text = np.random.choice(high_risk_texts)
            variations = [
                base_text,
                base_text + " I don't know how much longer I can take this.",
                base_text + " Something needs to change.",
                base_text + " I'm at my breaking point."
            ]
            texts.append(np.random.choice(variations))
            labels.append(2)
        
        return pd.DataFrame({'text': texts, 'label': labels})
    
    def extract_features(self, texts):
        """
        Extract comprehensive features from text
        """
        features = []
        
        for text in texts:
            # Clean text
            cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
            
            # Sentiment features
            sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Burnout keyword features (more nuanced)
            burnout_keywords = {
                'exhaustion': ['exhausted', 'tired', 'fatigued', 'drained'],
                'stress': ['stressed', 'overwhelmed', 'pressure', 'anxious'],
                'negative_emotions': ['frustrated', 'angry', 'irritable', 'hopeless'],
                'workload': ['overworked', 'busy', 'deadline', 'urgent'],
                'burnout_specific': ['burnt out', 'burnout', 'breaking point', 'can\'t take it']
            }
            
            keyword_counts = {}
            for category, keywords in burnout_keywords.items():
                keyword_counts[f'{category}_count'] = sum(1 for kw in keywords if kw in cleaned_text)
            
            # Text length features
            word_count = len(cleaned_text.split())
            
            # Combine all features
            feature_dict = {
                'negative_sentiment': sentiment_scores['neg'],
                'positive_sentiment': sentiment_scores['pos'],
                'compound_sentiment': sentiment_scores['compound'],
                'word_count': word_count,
                **keyword_counts
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def train(self, data=None):
        """
        Train the burnout detection model
        """
        if data is None:
            data = self.create_synthetic_dataset()
        
        print(f"Training on {len(data)} samples...")
        print(f"Label distribution: {data['label'].value_counts().to_dict()}")
        
        # Extract features
        features = self.extract_features(data['text'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, data['label'], test_size=0.2, random_state=42, stratify=data['label']
        )
        
        # Train model
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'Moderate Risk', 'High Risk']))
        
        self.is_trained = True
        return self.classifier.score(X_test, y_test)
    
    def predict(self, text):
        """
        Predict burnout level for given text
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_features([text])
        
        # Make prediction
        prediction = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)[0]
        
        # Map to risk levels
        risk_levels = ['Low Risk', 'Moderate Risk', 'High Risk']
        risk_level = risk_levels[prediction]
        confidence = max(probabilities)
        
        return {
            'risk_level': risk_level,
            'confidence': confidence,
            'probabilities': {
                'low_risk': probabilities[0],
                'moderate_risk': probabilities[1], 
                'high_risk': probabilities[2]
            },
            'explanation': self._explain_prediction(text, features.iloc[0])
        }
    
    def _explain_prediction(self, text, features):
        """
        Provide explanation for the prediction
        """
        explanations = []
        
        if features['negative_sentiment'] > 0.3:
            explanations.append("High negative sentiment detected")
        
        if features['exhaustion_count'] > 0:
            explanations.append(f"Exhaustion indicators found ({features['exhaustion_count']} mentions)")
        
        if features['stress_count'] > 0:
            explanations.append(f"Stress indicators found ({features['stress_count']} mentions)")
        
        if features['burnout_specific_count'] > 0:
            explanations.append(f"Direct burnout mentions found ({features['burnout_specific_count']} mentions)")
        
        if features['negative_emotions_count'] > 0:
            explanations.append(f"Negative emotions detected ({features['negative_emotions_count']} mentions)")
        
        if not explanations:
            explanations.append("No significant risk indicators detected")
        
        return explanations
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.is_trained:
            joblib.dump(self, filepath)
            print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model"""
        return joblib.load(filepath)

# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = ImprovedBurnoutDetector()
    
    # Train model
    accuracy = detector.train()
    print(f"\nModel accuracy: {accuracy:.2%}")
    
    # Test predictions
    test_cases = [
        "I love my job and feel energized every day!",
        "Work is busy but manageable. I'm a bit tired but okay.",
        "I'm completely exhausted and can't keep up anymore. I feel burnt out.",
        "The workload is reasonable and I have good support from my team.",
        "I'm overwhelmed with deadlines and feel stressed constantly."
    ]
    
    print("\n" + "="*50)
    print("TESTING PREDICTIONS")
    print("="*50)
    
    for text in test_cases:
        result = detector.predict(text)
        print(f"\nText: {text}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Explanation: {', '.join(result['explanation'])}")
    
    # Save model
    detector.save_model('improved_burnout_detector.pkl') 