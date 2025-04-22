import requests
import base64
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
GRAYLOG_URL = os.getenv("GRAYLOG_URL")
USERNAME = os.getenv("GRAYLOG_USERNAME")
PASSWORD = os.getenv("GRAYLOG_PASSWORD")
auth_str = f"{USERNAME}:{PASSWORD}"
headers = {
    "Authorization": "Basic " + base64.b64encode(auth_str.encode()).decode(),
    "Accept": "application/json",
}

def get_recent_logs(minutes=5):
    query_url = f"{GRAYLOG_URL}/search/universal/relative"
    params = {
        "query": "*",
        "range": minutes*60,
        "sort": "timestamp:desc",
        "fields": "timestamp,message,source,level",
        "limit": 10000
    }

    try:
        response = requests.get(query_url, headers=headers, params=params)
        messages = response.json()["messages"]
        
        # for rag
        return [m["message"] for m in messages]
    
        # for agentic rag
        # return [m["message"] for m in response.json()["messages"] if "message" in m]
        
    except Exception as e:
        print("Error fetching logs:", e)
        return []
# print(get_recent_logs())
