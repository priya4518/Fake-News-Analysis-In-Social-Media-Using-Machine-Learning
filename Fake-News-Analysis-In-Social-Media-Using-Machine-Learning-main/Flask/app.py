from flask import Flask, request, jsonify, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("C:\\Users\\kosan\\Downloads\\fake_news_detector_model.pkl")
vectorizer = joblib.load("C:\\Users\\kosan\\Downloads\\count_vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the news content from the form
        news = request.form['news_text']
        
        # Transform the text using the vectorizer
        news_vectorized = vectorizer.transform([news])
        
        # Make prediction
        prediction = model.predict(news_vectorized)
        
        # Output the result
        result = 'Fake News' if prediction[0] == 1 else 'Real News'
        
        return render_template('index.html', prediction_text=f'The news is: {result}')
    
if __name__ == "__main__":
    app.run(debug=True)
