from graylog_loader import get_recent_logs

# def get_logs(minutes=5):
#     """
#     Returns logs from the last `minutes` minutes from Graylog.
#     Default is 5 minutes if no argument is given.
#     """
#     try:
#         minutes = int(minutes)
#     except Exception:
#         minutes = 5  # fallback to default if invalid input
#     return get_recent_logs(minutes)
import pandas as pd
from typing import List, Dict, Union, Any

def get_logs(minutes=5):
    """
    Returns logs from the last `minutes` minutes.
    Default is 5 minutes if no argument is given.
    """
    try:
        minutes = int(minutes)
    except (ValueError, TypeError):
        minutes = 5
        
    df = pd.read_csv("logs.csv")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    now = pd.Timestamp.now(tz="UTC")
    time_window = now - pd.Timedelta(minutes=minutes)
    
    filtered = df[df["timestamp"] >= time_window]
    
    print(f"Current UTC time: {now}")
    print(f"Looking for logs after: {time_window}")
    print(f"Latest log timestamp: {df['timestamp'].max() if not df.empty else 'No logs'}")
    print(f"Found {len(filtered)} logs in the specified time window")
    
    result = filtered.to_dict(orient="records")
    if not result:
        return f"No logs found in the last {minutes} minutes. Latest log is from {df['timestamp'].max() if not df.empty else 'unknown'}."
    return result

def filter_critical(logs: Union[List[Dict], str]) -> Union[List[Dict], str]:
    """
    Filters logs that have a severity level of 4 or below.
    Returns a list of critical or error-level logs.
    """
    if isinstance(logs, str):
        return "Nothing to filter: " + logs
    
    critical_logs = [log for log in logs if log.get("level", 6) <= 4]
    if not critical_logs:
        return "No critical logs found."
    return critical_logs

def summarize_logs(logs: Union[List[Dict], str]) -> str:
    """
    Summarizes logs into a concise report.
    """
    if isinstance(logs, str):
        return "Cannot summarize: " + logs
    
    if not logs:
        return "No logs to summarize."

    lines = []
    for log in logs[:5]:
        time = log.get("timestamp", "Unknown time")
        source = log.get("source", "Unknown source")
        level = log.get("level", "Unknown level")
        msg = log.get("message", "No message")
        lines.append(f"[{time}] (Level: {level}, Source: {source}) {msg}")
    
    return "Summary of logs:\n" + "\n".join(lines)