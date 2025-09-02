# controllers.py
from fastapi import HTTPException
from typing import Dict, List
from fetcher import Fetcher

# Global flag to check if processing is done
processing_done = False  # Change to True when enrichment is complete

def set_processing_status(done: bool):
    """Set the global processing status."""
    global processing_done
    processing_done = done

def get_antisemitic_with_weapon() -> Dict:
    """
    Endpoint: Return antisemitic tweets with weapons if processing done.
    Else: return message.
    """
    global processing_done
    if not processing_done:
        return {"message": "Data is still being processed. Please wait."}

    fetcher = Fetcher()
    try:
        results = fetcher.get_antisemitic_with_weapon()
        return {"data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {e}")

def get_tweets_with_two_or_more_weapons() -> Dict:
    """
    Endpoint: Return tweets with 2 or more weapons if processing done.
    Else: return message.
    """
    global processing_done
    if not processing_done:
        return {"message": "Data is still being processed. Please wait."}

    fetcher = Fetcher()
    try:
        results = fetcher.get_tweets_with_two_or_more_weapons()
        return {"data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {e}")