import requests
import base64
from datetime import datetime, timedelta

# CONFIGURE YOUR GRAYLOG CREDENTIALS HERE:
GRAYLOG_URL = "http://10.181.4.5:9000/api"
USERNAME = "131hu5c77g2mr51khieu13t2j7sve9f6bdn5qnbijflgel6mpsl"
PASSWORD = "token"

# Prepare headers with basic auth
auth_str = f"{USERNAME}:{PASSWORD}"
headers = {
    "Authorization": "Basic " + base64.b64encode(auth_str.encode()).decode(),
    "Accept": "application/json",
}

def get_recent_logs(minutes=5):
    since = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat() + "Z"
    query_url = f"{GRAYLOG_URL}/search/universal/relative"
    params = {
        "query": "*",
        "range": minutes,
        "limit": 100,
        "sort": "timestamp:desc",
        "fields": "timestamp,message,source,level"
    }

    try:
        response = requests.get(query_url, headers=headers, params=params)
        messages = response.json()["messages"]
        return [m["message"] for m in messages]
    except Exception as e:
        print("Error fetching logs:", e)
        return []
