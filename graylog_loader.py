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
DEFAULT_CSV_FILE_PATH = "logs-24h-sorted.csv" 


DEFAULT_TIMEZONE = pytz.timezone('Europe/Belgrade')
REFERENCE_TIME = pytz.timezone('Europe/Belgrade').localize(datetime(2025, 5, 16, 12, 55, 0))

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

def get_logs_from_csv(file_path=None, max_logs=None, minutes_ago=None, hours_ago=None, 
                      start_time=None, end_time=None, source_ip=None):
    """
    Read logs from a CSV file with time filtering.
    """
    if file_path is None:
        file_path = DEFAULT_CSV_FILE_PATH
    
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
    logs_in_time_range = 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                logs_read += 1
                
                try:
                    log_time = parse_timestamp(row["timestamp"])
                    if log_time is None:
                        continue
                    
                    if logs_read <= 5:
                        print(f"DEBUG: Log {logs_read} - Raw: {row['timestamp']}, Parsed: {log_time}")
                    
                    time_match = True
                    if start_time and log_time < start_time:
                        time_match = False
                    if end_time and log_time > end_time:
                        time_match = False
                    
                    if time_match:
                        logs_in_time_range += 1
                        
                    # Apply source filter
                    if source_ip and row.get("source") != source_ip:
                        continue
                    
                    # Apply time filter
                    if not time_match:
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
                        print(f"DEBUG: Reached maximum log limit of {max_logs}")
                        break
                        
                except Exception as e:
                    print(f"DEBUG: Skipping row due to error: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
    
    print(f"DEBUG: Total logs read: {logs_read}")
    print(f"DEBUG: Logs in time range: {logs_in_time_range}")
    print(f"DEBUG: Final filtered logs: {logs_filtered}")
    
    logger.info(f"Loaded {logs_filtered} logs from CSV (filtered from {logs_read} total)")
    return logs

def get_logs_from_graylog(minutes=60, max_logs=50000, batch_size=5000):
    query_url = f"{GRAYLOG_URL}/search/universal/relative"
    collected_logs = []
    offset = 0

    while len(collected_logs) < max_logs:
        remaining = max_logs - len(collected_logs)
        current_batch = min(batch_size, remaining)

        params = {
            "query": "*",
            "range": minutes * 60,
            "sort": "timestamp:desc",
            "fields": "timestamp,message,source,level",
            "limit": current_batch,
            "offset": offset
        }

        try:
            logger.info(f"Fetching Graylog logs with offset={offset}, batch={current_batch}")
            response = requests.get(query_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            messages = data.get("messages", [])

            if not messages:
                break  # All done

            for m in messages:
                if "message" in m:
                    msg = m["message"]
                    try:
                        level = int(msg.get("level", 6))
                    except (ValueError, TypeError):
                        level = 6

                    collected_logs.append({
                        "timestamp": msg.get("timestamp", ""),
                        "source": msg.get("source", "Unknown"),
                        "message": msg.get("message", ""),
                        "level": level
                    })

            if len(messages) < current_batch:
                break

            offset += current_batch

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching logs from Graylog: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error fetching logs from Graylog: {e}")
            break

    logger.info(f"Total logs fetched from Graylog: {len(collected_logs)}")
    return collected_logs

def get_logs_from_graylog_absolute(from_time, to_time, max_logs=50000, batch_size=5000):
    query_url = f"{GRAYLOG_URL}/search/universal/absolute"
    collected_logs = []
    offset = 0

    from_str = from_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    to_str = to_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    while len(collected_logs) < max_logs:
        remaining = max_logs - len(collected_logs)
        current_batch = min(batch_size, remaining)

        params = {
            "query": "*",
            "from": from_str,
            "to": to_str,
            "fields": "timestamp,message,source,level",
            "limit": current_batch,
            "offset": offset
        }

        try:
            logger.info(f"Fetching Graylog logs from {from_str} to {to_str}, offset={offset}")
            response = requests.get(query_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            messages = data.get("messages", [])

            if not messages:
                logger.info("No more messages returned")
                break

            for m in messages:
                if "message" in m:
                    msg = m["message"]
                    try:
                        level = int(msg.get("level", 6))
                    except (ValueError, TypeError):
                        level = 6

                    collected_logs.append({
                        "timestamp": msg.get("timestamp", ""),
                        "source": msg.get("source", "Unknown"),
                        "message": msg.get("message", ""),
                        "level": level
                    })

            if len(messages) < current_batch:
                break

            offset += current_batch

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching logs from Graylog: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error fetching logs from Graylog: {e}")
            break

    logger.info(f"Total logs fetched from Graylog: {len(collected_logs)}")
    return collected_logs

# Main function that can use either csv or graylg api
def get_recent_logs(minutes=None, hours=None, max_logs=5000, use_csv=True, csv_path=None, 
                    start_time=None, end_time=None, source_ip=None):
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
            end_time=end_time,
            source_ip=source_ip
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