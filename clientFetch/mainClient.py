# main.py
from fastapi import FastAPI
import uvicorn
from controller import get_antisemitic_with_weapon, get_tweets_with_two_or_more_weapons, set_processing_status

app = FastAPI(title="Antisemitism Monitor API")

# --- Endpoints ---

@app.get("/antisemitic-with-weapon")
def antisemitic_with_weapon():
    """
    Return all antisemitic tweets that have at least one weapon.
    If processing is not done, return a waiting message.
    """
    return get_antisemitic_with_weapon()

@app.get("/two-or-more-weapons")
def two_or_more_weapons():
    """
    Return all tweets that have 2 or more weapons.
    If processing is not done, return a waiting message.
    """
    return get_tweets_with_two_or_more_weapons()

@app.post("/processing-done")
def mark_processing_done():
    """
    Call this endpoint when enrichment (weapons, sentiment) is complete.
    """
    set_processing_status(True)
    return {"status": "Processing marked as done. Endpoints are now active."}

@app.get("/")
def home():
    return {
        "message": "Welcome to the Antisemitism Monitoring API",
        "endpoints": [
            "GET /antisemitic-with-weapon",
            "GET /two-or-more-weapons",
            "POST /processing-done (mark enrichment complete)"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)