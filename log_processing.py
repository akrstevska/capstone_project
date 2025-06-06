"""
Log processing utilities to improve RAG retrieval
"""

import re
from collections import defaultdict
from datetime import datetime, timedelta
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

def extract_device_info(message):
    """
    Extract device type, ID, and other relevant information from log message.
    
    Args:
        message (str): The log message
        
    Returns:
        dict: Dictionary with extracted device information
    """
    device_info = {
        "device_type": None,
        "device_id": None,
        "pon": None,
        "onu_id": None,
        "mac": None,
        "ip": None
    }
    
    # OLT device info
    olt_match = re.search(r'(TK_AZ[_-]OLT[_-][A-Za-z0-9]+)', message)
    if olt_match:
        device_info["device_type"] = "OLT"
        device_info["device_id"] = olt_match.group(1)
    
    # ONU device info
    onu_match = re.search(r'\bONU[_\s](\d+)\b', message)
    if onu_match:
        device_info["device_type"] = "ONU"
        device_info["onu_id"] = onu_match.group(1)
    
    # PON info
    pon_match = re.search(r'PON\s+(\d+/\d+)', message)
    if pon_match:
        device_info["pon"] = pon_match.group(1)
    
    # MAC address
    mac_match = re.search(r'([0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2})', message)
    if mac_match:
        device_info["mac"] = mac_match.group(1)
        
    ip_match = re.search(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', message)
    if ip_match:
        device_info["ip"] = ip_match.group(1)
    
    return device_info

def enrich_log_metadata(log):
    """
    Enrich log metadata with extracted information for better retrieval.
    
    Args:
        log (dict): Log dictionary with timestamp, source, message, level
        
    Returns:
        dict: Enriched log dictionary with additional metadata
    """
    # device info
    device_info = extract_device_info(log.get("message", ""))
    
    enriched = log.copy()
    enriched.update({
        "device_type": device_info["device_type"],
        "device_id": device_info["device_id"],
        "pon": device_info["pon"],
        "onu_id": device_info["onu_id"],
        "mac": device_info["mac"],
        "ip": device_info["ip"]
    })
    
    return enriched

def logs_to_chunked_documents(logs, chunk_size=5, chunk_overlap=1):
    """
    Convert logs to chunked documents grouped by source for better retrieval.
    
    Args:
        logs (list): List of log dictionaries
        chunk_size (int): Number of logs per document chunk
        chunk_overlap (int): Number of logs to overlap between chunks
        
    Returns:
        list: List of langchain Document objects
    """
    # Group logs by source
    source_logs = defaultdict(list)
    for log in logs:
        source = log.get("source", "Unknown")
        source_logs[source].append(log)
    
    documents = []
    
    for source, logs_list in source_logs.items():
        for log in logs_list:
            enriched_log = enrich_log_metadata(log)
            
            doc = Document(
                page_content=log.get("message", ""),
                metadata={
                    "source": source,
                    "timestamp": log.get("timestamp", ""),
                    "datetime": log.get("datetime"), 
                    "level": log.get("level", 6),
                    "device_type": enriched_log.get("device_type"),
                    "device_id": enriched_log.get("device_id"),
                    "pon": enriched_log.get("pon"),
                    "onu_id": enriched_log.get("onu_id"),
                    "mac": enriched_log.get("mac"),
                    "ip": enriched_log.get("ip")
                }
            )
            documents.append(doc)
    
    return documents

def create_enhanced_vector_store(logs, embedding_model):
    """
    Create an enhanced vector store with better metadata for improved retrieval.
    
    Args:
        logs (list): List of log dictionaries
        embedding_model: Embedding model for vectorization
        
    Returns:
        FAISS: Vector store with enhanced documents
    """
    documents = logs_to_chunked_documents(logs)
    
    vector_store = FAISS.from_documents(documents, embedding_model)
    
    return vector_store

def extract_query_filters(query):
    """
    Extract filter parameters from a user query.
    
    Args:
        query (str): User query
        
    Returns:
        dict: Dictionary with extracted filters
    """
    filters = {}
    
    # Device type filter
    if re.search(r'\bOLT\b', query, re.IGNORECASE):
        filters["device_type"] = "OLT"
    elif re.search(r'\bONU\b', query, re.IGNORECASE):
        filters["device_type"] = "ONU"
    
    # Device ID filter
    olt_match = re.search(r'(TK_AZ[_-]OLT[_-][A-Za-z0-9]+)', query)
    if olt_match:
        filters["device_id"] = olt_match.group(1)
    
    # PON filter
    pon_match = re.search(r'PON\s+(\d+/\d+)', query)
    if pon_match:
        filters["pon"] = pon_match.group(1)
    
    # ONU ID filter
    onu_match = re.search(r'ONU[_\s](\d+)', query)
    if onu_match:
        filters["onu_id"] = onu_match.group(1)
    
    # MAC address filter
    mac_match = re.search(r'([0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2})', query)
    if mac_match:
        filters["mac"] = mac_match.group(1)
    
    # IP filter - FIXED VERSION
    ip_match = re.search(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', query)
    if ip_match:
        filters["target_ip"] = ip_match.group(1)  # Use unique key to avoid conflicts
    
    # Level filter (critical, warning...)
    if re.search(r'\bcritical\b|\berror\b|\bfatal\b', query, re.IGNORECASE):
        filters["max_level"] = 2  # show levels 0-2 (fatal, critical, error)
    elif re.search(r'\bwarning\b', query, re.IGNORECASE):
        filters["max_level"] = 3  # show levels 0-3 (fatal, critical, error, warning)
    
    # Time filters
    time_filters = extract_time_filters(query)
    if time_filters:
        filters.update(time_filters)
    
    return filters

def extract_time_filters(query):
    """
    Extract time-related filters from a user query.
    
    Args:
        query (str): User query
        
    Returns:
        dict: Dictionary with time filters
    """
    time_filters = {}
    
    # Last X minutes
    minutes_match = re.search(r'last\s+(\d+)\s+minutes?', query, re.IGNORECASE)
    if minutes_match:
        time_filters["minutes_ago"] = int(minutes_match.group(1))
        return time_filters
    
    # Last X hours
    hours_match = re.search(r'last\s+(\d+)\s+hours?', query, re.IGNORECASE)
    if hours_match:
        time_filters["hours_ago"] = int(hours_match.group(1))
        return time_filters
    
    # Last hour
    if re.search(r'last\s+hour', query, re.IGNORECASE):
        time_filters["hours_ago"] = 1
        return time_filters
    
    # Today
    if re.search(r'\btoday\b', query, re.IGNORECASE):
        time_filters["today"] = True
        return time_filters
    
    # Yesterday
    if re.search(r'\byesterday\b', query, re.IGNORECASE):
        time_filters["yesterday"] = True
        return time_filters
    
    return time_filters

def create_filter_function(filters):
    """
    Create a filter function for vector store retrieval based on filters.
    
    Args:
        filters (dict): Dictionary of filters to apply
        
    Returns:
        function: Filter function for vector store retrieval
    """
    def filter_function(doc):
        metadata = getattr(doc, "metadata", None)
        if not isinstance(metadata, dict):
            return False

        for key, value in filters.items():
            if key == "minutes_ago" and "datetime" in metadata:
                from graylog_loader import get_reference_time
                ref_time = get_reference_time()
                doc_time = metadata["datetime"]
                if not doc_time or doc_time < (ref_time - timedelta(minutes=value)):
                    return False
            elif key == "hours_ago" and "datetime" in metadata:
                from graylog_loader import get_reference_time
                ref_time = get_reference_time()
                doc_time = metadata["datetime"]
                if not doc_time or doc_time < (ref_time - timedelta(hours=value)):
                    return False
            elif key == "today" and "datetime" in metadata:
                from graylog_loader import get_reference_time
                ref_time = get_reference_time()
                doc_time = metadata["datetime"]
                if not doc_time or doc_time.date() != ref_time.date():
                    return False
            elif key == "yesterday" and "datetime" in metadata:
                from graylog_loader import get_reference_time
                ref_time = get_reference_time()
                yesterday = (ref_time - timedelta(days=1)).date()
                doc_time = metadata["datetime"]
                if not doc_time or doc_time.date() != yesterday:
                    return False
            elif key == "max_level":
                doc_level = metadata.get("level", 6)
                if doc_level > value:
                    return False
            elif key == "target_ip":
                doc_source = metadata.get("source", "")
                doc_ip = metadata.get("ip", "")
                
                ip_found = (str(doc_source).strip() == str(value).strip() or 
                           str(doc_ip).strip() == str(value).strip())
                
                if not ip_found:
                    return False
            elif key in metadata:
                doc_value = metadata[key]
                if doc_value is None:
                    return False
                if key in ["onu_id", "device_id", "pon", "source"]:
                    if str(doc_value).strip() != str(value).strip():
                        return False
                elif isinstance(doc_value, str) and isinstance(value, str):
                    if value.lower() not in str(doc_value).lower():
                        return False
                elif doc_value != value:
                    return False
        return True

    return filter_function