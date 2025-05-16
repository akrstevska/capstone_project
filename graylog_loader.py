import requests
import base64
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
import os
import csv
import logging
# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('graylog_loader')

load_dotenv()
GRAYLOG_URL = os.getenv("GRAYLOG_URL")
USERNAME = os.getenv("GRAYLOG_USERNAME")
PASSWORD = os.getenv("GRAYLOG_PASSWORD")
auth_str = f"{USERNAME}:{PASSWORD}"
headers = {
    "Authorization": "Basic " + base64.b64encode(auth_str.encode()).decode(),
    "Accept": "application/json",
}
DEFAULT_CSV_FILE_PATH = "logs-24h.csv" 


DEFAULT_TIMEZONE = pytz.timezone('Europe/Belgrade')
REFERENCE_TIME = pytz.timezone('Europe/Belgrade').localize(datetime(2025, 5, 16, 10, 15, 0))

def set_reference_time(time_str=None):
    return REFERENCE_TIME

def get_reference_time():
    return REFERENCE_TIME

def parse_timestamp(timestamp_str):
    """
    Parse timestamp string to datetime object, handling different formats.
    
    Args:
        timestamp_str (str): Timestamp string
        
    Returns:
        datetime: Parsed datetime object with timezone
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",  
        "%Y-%m-%dT%H:%M:%SZ",     
        "%Y-%m-%d %H:%M:%S",      
        "%Y-%m-%d %H:%M:%S.%f"     
    ]
    
    dt = None
    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            break
        except ValueError:
            continue
    
    if dt is None:
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return None
    
    # timezone if missing
    if dt.tzinfo is None:
        dt = DEFAULT_TIMEZONE.localize(dt)
    
    return dt

# read from CSV file 
def get_logs_from_csv(file_path=None, max_logs=None, minutes_ago=None, hours_ago=None, start_time=None, end_time=None):
    """
    Read logs from a CSV file with time filtering.
    
    Args:
        file_path (str): Path to the CSV file. Defaults to DEFAULT_CSV_FILE_PATH.
        max_logs (int): Maximum number of logs to read. If None, read all logs.
        minutes_ago (int): Filter logs from the last X minutes from reference time.
        hours_ago (int): Filter logs from the last X hours from reference time.
        start_time (datetime): Start time for filtering logs.
        end_time (datetime): End time for filtering logs.
        
    Returns:
        list: List of log dictionaries
    """
    if file_path is None:
        file_path = DEFAULT_CSV_FILE_PATH
    
    # Determine time filter
    ref_time = get_reference_time()
    
    if minutes_ago is not None:
        start_time = ref_time - timedelta(minutes=minutes_ago)
        end_time = ref_time
    elif hours_ago is not None:
        start_time = ref_time - timedelta(hours=hours_ago)
        end_time = ref_time
    
    logs = []
    logs_read = 0
    logs_filtered = 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                logs_read += 1
                
                try:
                    log_time = parse_timestamp(row["timestamp"])
                    if log_time is None:
                        continue
                    
                    if start_time and log_time < start_time:
                        continue
                    if end_time and log_time > end_time:
                        continue
                    
                    try:
                        level = int(row["level"])
                    except (ValueError, TypeError):
                        level = 6 
                    
                    logs.append({
                        "timestamp": row["timestamp"],
                        "datetime": log_time,  
                        "source": row["source"],
                        "message": row["message"],
                        "level": level,
                    })
                    logs_filtered += 1
                    
                    if max_logs is not None and logs_filtered >= max_logs:
                        logger.info(f"Reached maximum log limit of {max_logs}")
                        break
                        
                except Exception as e:
                    logger.warning(f"Skipping row due to error: {e}")
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
    
    logger.info(f"Loaded {logs_filtered} logs from CSV (filtered from {logs_read} total)")
    return logs

# get logs from Graylog API - next milestone
def get_logs_from_graylog(minutes=60, max_logs=10000):
    """
    Get logs from Graylog API.
    
    Args:
        minutes (int): Number of minutes to look back.
        max_logs (int): Maximum number of logs to retrieve.
        
    Returns:
        list: List of log dictionaries
    """
    query_url = f"{GRAYLOG_URL}/search/universal/relative"
    params = {
        "query": "*",
        "range": minutes*60,
        "sort": "timestamp:desc",
        "fields": "timestamp,message,source,level",
        "limit": max_logs
    }

    try:
        logger.info(f"Fetching logs from Graylog (past {minutes} minutes)")
        response = requests.get(query_url, headers=headers, params=params)
        response.raise_for_status()  
        
        messages = response.json().get("messages", [])
        
        logs = []
        for m in messages:
            if "message" in m:
                msg = m.get("message", {})
                
                log_time = parse_timestamp(msg.get("timestamp", ""))
                
                logs.append({
                    "timestamp": msg.get("timestamp", ""),
                    "datetime": log_time,
                    "source": msg.get("source", "Unknown"),
                    "message": msg.get("message", ""),
                    "level": int(msg.get("level", 6)), 
                })
        
        logger.info(f"Fetched {len(logs)} logs from Graylog")
        return logs
    
    except Exception as e:
        logger.error(f"Error fetching logs from Graylog: {e}")
        return []

# Main function that can use either csv or graylg api
def get_recent_logs(minutes=None, hours=None, max_logs=5000, use_csv=True, csv_path=None, 
                    start_time=None, end_time=None):
    """
    Get recent logs from either CSV file or Graylog API with flexible time filtering.
    
    Args:
        minutes (int): Number of minutes to look back from reference time.
        hours (int): Number of hours to look back from reference time.
        max_logs (int): Maximum number of logs to retrieve.
        use_csv (bool): Whether to use CSV file instead of Graylog API.
        csv_path (str): Path to the CSV file. If None, use DEFAULT_CSV_FILE_PATH.
        start_time (datetime): Start time for filtering logs.
        end_time (datetime): End time for filtering logs.
        
    Returns:
        list: List of log dictionaries
    """
    if use_csv:
        return get_logs_from_csv(
            file_path=csv_path, 
            max_logs=max_logs, 
            minutes_ago=minutes, 
            hours_ago=hours,
            start_time=start_time,
            end_time=end_time
        )
    else:
        if hours is not None:
            minutes = hours * 60
        elif minutes is None:
            minutes = 60  # default to 1 hour
            
        return get_logs_from_graylog(minutes=minutes, max_logs=max_logs)

if __name__ == "__main__":
    # test
    set_reference_time("2023-05-15T14:00:00")
    
    logs = get_recent_logs(hours=1, max_logs=10)
    print(f"Found {len(logs)} logs in the last hour")
    
    for log in logs[:3]:
        print(log)