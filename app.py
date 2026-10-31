import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import joblib
import pandas as pd
import os
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql

# PyMySQL ko MySQLdb driver ki tarah register karta hai
pymysql.install_as_MySQLdb()

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "manomitra_secret_key_123")
CORS(app)

# --- 1. MySQL Database Configuration ---
# Apne local MySQL credentials fill karein (ya .env me add karein)
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "your_mysql_password")  # <--- Apna MySQL password yahan daalein
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "mental_health_db")

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_POOL_RECYCLE'] = 280  # MySQL timeout prevent karta hai
db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    assessments = db.relationship('Assessment', backref='user', lazy=True)

class Assessment(db.Model):
    __tablename__ = 'assessments'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    wellbeing_score = db.Column(db.Integer)

with app.app_context():
    db.create_all()
    # 🧹 Auto-Purge: Server restart par zero/null entries auto clean hongi
    try:
        bad_records = Assessment.query.filter(
            (Assessment.wellbeing_score == 0) | 
            (Assessment.wellbeing_score == None)
        ).all()
        for rec in bad_records:
            db.session.delete(rec)
        db.session.commit()
        if bad_records:
            print(f"🧹 Purged {len(bad_records)} bad zero-score records from MySQL.")
    except Exception as e:
        print(f"Database sync note: {e}")

# --- 2. Load ML Model ---
MODEL_PATH = 'mental_health_model.joblib'
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ AI MODEL LOADED SUCCESSFULLY!")
    except Exception as e:
        print(f"⚠️ ERROR LOADING MODEL: {e}")

# --- 3. Groq Chatbot ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def get_today_start():
    now = datetime.now()
    return datetime(now.year, now.month, now.day, 0, 0, 0)


# --- AUTHENTICATION ROUTES ---

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json() or {}
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not name or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"error": "An account with this email already exists."}), 400

    hashed_pw = generate_password_hash(password)
    new_user = User(name=name, email=email, password_hash=hashed_pw)
    
    db.session.add(new_user)
    db.session.commit()

    session['user_id'] = new_user.id
    session['user_name'] = new_user.name
    session.permanent = True

    return jsonify({"status": "success", "name": new_user.name})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password."}), 401

    session['user_id'] = user.id
    session['user_name'] = user.name
    session.permanent = True

    return jsonify({"status": "success", "name": user.name})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"status": "success"})

@app.route('/api/me', methods=['GET'])
def check_session():
    if 'user_id' in session:
        return jsonify({"logged_in": True, "name": session['user_name']})
    return jsonify({"logged_in": False})


# --- CORE APP ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        raw_score = data.get('wellbeing_score', 50)
        
        try:
            w_score = int(raw_score)
            if w_score <= 0:
                w_score = 50
        except:
            w_score = 50

        user_id = session.get('user_id')
        prediction_str = "Completed"
        confidence_val = 1.0

        answers_dict = data.get('answers', {})
        if model is not None and isinstance(answers_dict, dict) and len(answers_dict) > 0:
            try:
                input_df = pd.DataFrame([answers_dict])
                prediction_val = model.predict(input_df)[0]
                prediction_str = str(prediction_val)
                probabilities = model.predict_proba(input_df)[0].tolist()
                confidence_val = max(probabilities)
            except Exception as ml_err:
                print(f"ML warning: {ml_err}")

        today_start = get_today_start()
        
        if user_id:
            today_record = Assessment.query.filter(
                Assessment.user_id == user_id,
                Assessment.timestamp >= today_start
            ).first()
        else:
            today_record = Assessment.query.filter(
                Assessment.timestamp >= today_start
            ).first()

        if today_record:
            today_record.wellbeing_score = w_score
            today_record.prediction = prediction_str
            today_record.confidence = confidence_val
            today_record.timestamp = datetime.now()
        else:
            new_record = Assessment(
                user_id=user_id,
                prediction=prediction_str, 
                confidence=confidence_val,
                wellbeing_score=w_score
            )
            db.session.add(new_record)

        db.session.commit()

        if user_id:
            records = Assessment.query.filter(
                Assessment.user_id == user_id,
                Assessment.wellbeing_score > 0
            ).all()
        else:
            records = Assessment.query.filter(Assessment.wellbeing_score > 0).all()

        valid_scores = [r.wellbeing_score for r in records if r.wellbeing_score is not None]
        overall_score = int(sum(valid_scores) / len(valid_scores)) if valid_scores else w_score

        return jsonify({
            "status": "success", 
            "score": overall_score,
            "prediction": prediction_str
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/daily-score', methods=['GET'])
def get_daily_score():
    try:
        user_id = session.get('user_id')
        today_start = get_today_start()
        
        if user_id:
            all_records = Assessment.query.filter(
                Assessment.user_id == user_id,
                Assessment.wellbeing_score > 0
            ).all()
            today_record = Assessment.query.filter(
                Assessment.user_id == user_id,
                Assessment.timestamp >= today_start
            ).first()
        else:
            all_records = Assessment.query.filter(Assessment.wellbeing_score > 0).all()
            today_record = Assessment.query.filter(Assessment.timestamp >= today_start).first()

        valid_scores = [r.wellbeing_score for r in all_records if r.wellbeing_score is not None]
        overall_score = int(sum(valid_scores) / len(valid_scores)) if valid_scores else 0
        has_taken_today = today_record is not None

        status_msg = f"Overall score based on {len(valid_scores)} daily check-in(s)." if valid_scores else "Take a survey check-in to initialize tracking."

        return jsonify({
            "score": overall_score,
            "has_taken_today": has_taken_today,
            "today_score": today_record.wellbeing_score if today_record else None,
            "status": status_msg
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if not client:
        return jsonify({"reply": "Chatbot API key is missing."}), 500

    try:
        data = request.get_json() or {}
        user_message = data.get("message")

        if not user_message:
            return jsonify({"reply": "Please enter a message."}), 400

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are Manomitra, a compassionate mental health companion. Keep replies short, warm, safe, and conversational."
                },
                {"role": "user", "content": user_message}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.7,
            max_tokens=500
        )
        
        return jsonify({"status": "success", "reply": chat_completion.choices[0].message.content})
    except Exception as e:
        print(f"❌ GROQ ERROR: {e}")
        return jsonify({"reply": "Connection issue. Please take a deep breath and try sending again."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)