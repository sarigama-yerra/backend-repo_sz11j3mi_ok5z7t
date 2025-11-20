import os
from typing import Optional, Literal, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="AI Behavior Analysis Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BehaviorInput(BaseModel):
    facial_expression: Optional[Literal[
        "neutral", "happy", "sad", "angry", "surprised", "fearful", "disgust", "contempt"
    ]] = Field(default=None, description="Primary detected facial expression")
    eye_movement: Optional[Literal[
        "steady", "frequent_blinks", "rapid_saccades", "averted_gaze", "downcast", "closed"
    ]] = None
    head_tilt: Optional[Literal["neutral", "left", "right", "forward", "back"]] = None
    upper_body_movement: Optional[Literal[
        "still", "restless", "open_posture", "closed_posture", "fidgeting"
    ]] = None
    voice_tone: Optional[Literal[
        "calm", "tense", "shaky", "loud", "soft", "monotone", "varied"
    ]] = None
    text_message: Optional[str] = Field(default=None, description="Optional text transcript or message")

    # Optional numeric signals (if available)
    blink_rate: Optional[float] = Field(default=None, ge=0, le=100, description="blinks/min")
    gaze_stability: Optional[float] = Field(default=None, ge=0, le=1, description="0-1 where 1 is very stable gaze")
    speech_rate_wpm: Optional[float] = Field(default=None, ge=0, le=400)
    pitch_variability: Optional[float] = Field(default=None, ge=0, le=1)


class AnalysisResult(BaseModel):
    emotion_percentages: Dict[str, int]
    stress_score: int
    attention_score: int
    confidence_level: int
    honesty_probability: int
    summary: str
    suggestions: Dict[str, Any]


def clamp(v: float, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(round(v))))


def analyze_behavior(data: BehaviorInput) -> AnalysisResult:
    # Base priors
    emotions = {
        "neutral": 20,
        "happy": 15,
        "sad": 10,
        "angry": 8,
        "surprised": 10,
        "fearful": 8,
        "disgust": 5,
        "contempt": 5,
    }

    stress = 30.0
    attention = 60.0
    confidence = 55.0
    honesty = 65.0

    # Facial expression influence
    if data.facial_expression:
        fe = data.facial_expression
        for k in emotions:
            emotions[k] *= 0.7  # dampen others
        emotions[fe] = emotions.get(fe, 10) + 35
        if fe == "happy":
            stress -= 10; confidence += 10; honesty += 5
        elif fe in ["sad", "fearful"]:
            stress += 10; confidence -= 10
        elif fe in ["angry", "disgust", "contempt"]:
            stress += 12; honesty -= 5
        elif fe == "surprised":
            attention += 5

    # Eyes
    if data.eye_movement:
        em = data.eye_movement
        if em == "steady":
            attention += 10; honesty += 5
        if em == "frequent_blinks":
            stress += 12; attention -= 8
        if em == "rapid_saccades":
            stress += 8; attention -= 6
        if em in ["averted_gaze", "downcast"]:
            confidence -= 8; honesty -= 6
        if em == "closed":
            attention -= 15

    # Head tilt
    if data.head_tilt:
        ht = data.head_tilt
        if ht == "neutral":
            confidence += 4
        if ht in ["left", "right"]:
            attention += 2  # inquisitive
        if ht in ["forward"]:
            attention += 6; confidence += 3
        if ht in ["back"]:
            confidence -= 6

    # Upper body
    if data.upper_body_movement:
        ub = data.upper_body_movement
        if ub == "still":
            attention += 6
        if ub in ["restless", "fidgeting"]:
            stress += 12; attention -= 6
        if ub == "open_posture":
            confidence += 10; honesty += 6
        if ub == "closed_posture":
            confidence -= 10; honesty -= 6

    # Voice tone
    if data.voice_tone:
        vt = data.voice_tone
        if vt == "calm":
            stress -= 12; confidence += 8; honesty += 4
        if vt == "tense":
            stress += 15; confidence -= 6
        if vt == "shaky":
            stress += 10; confidence -= 10; honesty -= 4
        if vt == "loud":
            confidence += 6; honesty -= 3
        if vt == "soft":
            confidence -= 6
        if vt == "monotone":
            attention -= 6
        if vt == "varied":
            attention += 8

    # Numeric modifiers
    if data.blink_rate is not None:
        # Typical blink ~ 15-20/min
        stress += (data.blink_rate - 18) * 0.8
    if data.gaze_stability is not None:
        attention += (data.gaze_stability - 0.5) * 40
    if data.speech_rate_wpm is not None:
        # Normal conversational 140-180
        diff = data.speech_rate_wpm - 160
        if abs(diff) < 20:
            confidence += 4
        else:
            stress += min(15, abs(diff) * 0.2)
            confidence -= min(10, abs(diff) * 0.1)
    if data.pitch_variability is not None:
        # More variability -> engagement
        attention += (data.pitch_variability - 0.5) * 30

    # Text sentiment heuristic
    if data.text_message:
        text = data.text_message.lower()
        positive_words = ["excited", "happy", "confident", "great", "love", "glad", "thanks"]
        negative_words = ["nervous", "worried", "anxious", "hate", "upset", "sorry"]
        honesty_cues = ["honest", "truth", "frankly", "to be honest"]
        if any(w in text for w in positive_words):
            emotions["happy"] = emotions.get("happy", 10) + 10
            stress -= 6; confidence += 6
        if any(w in text for w in negative_words):
            emotions["sad"] = emotions.get("sad", 10) + 10
            stress += 6; confidence -= 6
        if any(w in text for w in honesty_cues):
            honesty += 4

    # Normalize emotion percentages to 100
    total = sum(max(v, 0) for v in emotions.values()) or 1
    emotion_percentages = {k: clamp(v / total * 100) for k, v in emotions.items()}

    stress_score = clamp(stress)
    attention_score = clamp(attention)
    confidence_level = clamp(confidence)
    honesty_probability = clamp(honesty)

    # Build summary
    top_emotion = max(emotion_percentages, key=emotion_percentages.get)
    summary_parts = [
        f"Dominant emotion appears to be '{top_emotion}'.",
        f"Stress is estimated at {stress_score}/100 with attention at {attention_score}/100.",
        f"Confidence around {confidence_level}/100 and honesty likelihood about {honesty_probability}/100.",
    ]

    # Suggestions
    suggestions = {
        "interviewer": [],
        "teacher": [],
        "user": [],
    }
    # Interviewer tips
    if stress_score > 65:
        suggestions["interviewer"].append("Reduce pressure with an easy, open question and slower pacing.")
    if attention_score < 45:
        suggestions["interviewer"].append("Re-engage with a concise prompt and maintain eye contact.")
    if confidence_level < 50:
        suggestions["interviewer"].append("Acknowledge positives to build confidence before deep questions.")

    # Teacher tips
    if attention_score < 55:
        suggestions["teacher"].append("Break content into smaller chunks and check for understanding.")
    if top_emotion in ["sad", "fearful", "angry"]:
        suggestions["teacher"].append("Use supportive tone and allow short reflection time.")

    # User tips
    if stress_score > 60:
        suggestions["user"].append("Try a 4-7-8 breath cycle to lower arousal levels.")
    if confidence_level < 55:
        suggestions["user"].append("Adopt an open posture and steady pace. Practice your key points.")
    if honesty_probability < 55:
        suggestions["user"].append("Answer directly and avoid over-qualifying statements.")

    return AnalysisResult(
        emotion_percentages=emotion_percentages,
        stress_score=stress_score,
        attention_score=attention_score,
        confidence_level=confidence_level,
        honesty_probability=honesty_probability,
        summary=" ".join(summary_parts),
        suggestions=suggestions,
    )


@app.post("/analyze", response_model=AnalysisResult)
def analyze_endpoint(payload: BehaviorInput):
    return analyze_behavior(payload)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
