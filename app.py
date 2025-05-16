from flask import Flask, request, jsonify
from flask_cors import CORS 

from graylog_loader import get_recent_logs, set_reference_time
from nlplog_utils import load_nlplog_vectorstore
from prompt_templates import (
    short_summary_prompt,
    detailed_analysis_prompt,
    critical_events_prompt,
    report_generator_prompt
)
import log_processing as lp

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.schema import Document
import re
from collections import defaultdict, Counter
import logging
from datetime import datetime

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('app')

app = Flask(__name__)
CORS(app) 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
nlplog_store = load_nlplog_vectorstore(embedding_model)
vector_store = None

def normalize_message(message):
    """
    Normalize log message by replacing variable parts.
    """
    # MAC addresses
    normalized = re.sub(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", "XX:XX:XX:XX:XX:XX", message)
    
    # PON identifiers
    normalized = re.sub(r'\bPON \d+/\d+\b', 'PON X/X', normalized)
    
    # ONU identifiers
    normalized = re.sub(r'\bONU \d+\b', 'ONU X', normalized)
    
    # IP addresses
    normalized = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'X.X.X.X', normalized)
    
    # timestamps
    normalized = re.sub(r'\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?\b', 'YYYY-MM-DD HH:MM:SS', normalized)
    
    return normalized

def cluster_logs_by_source(logs):
    """
    Group logs by source device and similar messages (removing variable parts).
    
    Args:
        logs (list): List of log dictionaries with 'source', 'message', and 'timestamp' keys
        
    Returns:
        dict: Dictionary with sources as keys and lists of message summaries as values
    """
    source_clusters = defaultdict(Counter)
    
    for log in logs:
        message = log.get("message", "")
        source = log.get("source", "Unknown")
        
        normalized_msg = normalize_message(message)
        
        source_clusters[source][normalized_msg] += 1
    
    summarized = {}
    for source, counter in source_clusters.items():
        summarized[source] = [
            f'"{msg}" occurred {count} times'
            for msg, count in counter.most_common()
        ]
    
    return summarized

@app.route("/clustered-logs", methods=["GET"])
def clustered_logs():
    try:
        hours = request.args.get('hours', type=int)
        minutes = request.args.get('minutes', type=int)
        reference_time = request.args.get('reference_time')
        
        if reference_time:
            set_reference_time(reference_time)
            
        logs = get_recent_logs(
            max_logs=5000,
            hours=hours,
            minutes=minutes
        )
        
        if not logs:
            return jsonify({"summary": "No logs available."})
        
        # important logs (level < 4)
        important_logs = [log for log in logs if log.get("level", 6) < 4]
        
        # Group by source
        grouped = cluster_logs_by_source(important_logs)
        return jsonify(grouped)
    except Exception as e:
        logger.error(f"Error in clustered-logs: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_q = request.json.get("question")
        style = request.json.get("style", "summary")
        
        reference_time = request.json.get("reference_time")
        if reference_time:
            set_reference_time(reference_time)
        
        filters = lp.extract_query_filters(user_q)
        time_filters = {k: v for k, v in filters.items() 
                       if k in ['minutes_ago', 'hours_ago', 'today', 'yesterday']}
        logger.info(f"Extracted time filters: {time_filters}")
        
        logs = get_recent_logs(
            max_logs=20000,
            minutes=time_filters.get('minutes_ago'),
            hours=time_filters.get('hours_ago')
            
        )
        
        if not logs:
            return jsonify({"answer": "No logs found for the specified time period."})

        k_logs = 20  
        if style == "detailed":
            k_logs = 40
        elif style == "critical":
            k_logs = 50
        elif style == "report":
            k_logs = 50

        if style == "summary":
            system_prompt = short_summary_prompt(user_q)
        elif style == "detailed":
            system_prompt = detailed_analysis_prompt(user_q)
        elif style == "critical":
            system_prompt = critical_events_prompt(user_q)
        elif style == "report":
            system_prompt = report_generator_prompt(user_q)
        else:
            system_prompt = short_summary_prompt(user_q)
            
        vector_store = lp.create_enhanced_vector_store(logs, embedding_model)
        
        llm = OllamaLLM(model="llama3")
        
        if filters:
            filter_function = lp.create_filter_function(filters)
            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": k_logs,
                    "filter": filter_function
                }
            )
        else:
            retriever = vector_store.as_retriever(search_kwargs={"k": k_logs})
        
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        try:
            response = qa.invoke(system_prompt)
            if isinstance(response, dict) and "result" in response:
                answer = response["result"]
            else:
                answer = qa.run(system_prompt)
                logger.info("Used legacy run() method as fallback")
        except Exception as e:
            logger.warning(f"Error with invoke method: {e}")
            answer = qa.run(system_prompt) 
            logger.info("Used legacy run() method as fallback")

        relevant_logs = []
        try:
           
            relevant_docs = vector_store.similarity_search(user_q, k=k_logs)
            
            for doc in relevant_docs:
                try:
                    metadata = {}
                    content = ""
                    
                    if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                        metadata = doc.metadata
                        content = doc.page_content
                    elif isinstance(doc, dict):
                        metadata = doc.get('metadata', {})
                        content = doc.get('page_content', str(doc))
                    else:
                        content = str(doc)
                    
                    source = metadata.get('source', 'Unknown')
                    level = metadata.get('level', 'Unknown')
                    timestamp = metadata.get('timestamp', 'Unknown')
                    device_type = metadata.get('device_type', None)
                    device_id = metadata.get('device_id', 'Unknown')
                    onu_id = metadata.get('onu_id', None)
                    pon = metadata.get('pon', None)
                    
                    metadata_parts = [
                        f"Source: {source}",
                        f"Level: {level}",
                        f"Time: {timestamp}"
                    ]
                    
                    if device_type == "OLT" and device_id:
                        metadata_parts.append(f"Device: {device_id}")
                    elif device_type == "ONU":
                        if onu_id and pon:
                            metadata_parts.append(f"ONU {onu_id} on PON {pon}")
                        elif onu_id:
                            metadata_parts.append(f"ONU {onu_id}")
                    
                    metadata_str = " | ".join(metadata_parts)
                    formatted_log = f"[{metadata_str}] {content}"
                    relevant_logs.append(formatted_log)
                    
                except Exception as e:
                    logger.warning(f"Error formatting individual log: {e}")
                    relevant_logs.append(f"[Raw Log] {str(doc)}")
                    
        except Exception as e:
            logger.error(f"Error retrieving or processing relevant logs: {e}")
            relevant_logs = ["Error retrieving supporting logs"]

        # result
        return jsonify({
            "answer": answer,
            "supporting_logs": relevant_logs,
            "metadata": {
                "retrieved_count": len(relevant_logs),
                "applied_filters": filters,
                "total_logs_searched": len(logs),
                "time_filters_applied": time_filters
            }
        })
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route("/stats", methods=["GET"])
def stats():
    """Get statistics about the logs"""
    try:
        hours = request.args.get('hours', type=int)
        minutes = request.args.get('minutes', type=int)
        reference_time = request.args.get('reference_time')
        
        if reference_time:
            set_reference_time(reference_time)
            
        logs = get_recent_logs(
            max_logs=20000,
            hours=hours,
            minutes=minutes
        )
        
        # logs by level
        level_counts = Counter([log.get("level", 6) for log in logs])
        
        # logs by source
        source_counts = Counter([log.get("source", "Unknown") for log in logs])
        
        # OLT and ONU devices
        devices = set()
        for log in logs:
            message = log.get("message", "")
            device_info = lp.extract_device_info(message)
            if device_info["device_id"]:
                devices.add(device_info["device_id"])
        
        time_range = "all available logs"
        if hours:
            time_range = f"last {hours} hours"
        elif minutes:
            time_range = f"last {minutes} minutes"
        
        return jsonify({
            "total_logs": len(logs),
            "logs_by_level": {str(k): v for k, v in level_counts.items()},
            "logs_by_source": dict(source_counts.most_common(10)),
            "unique_devices": len(devices),
            "sample_devices": list(devices)[:10] if devices else [],
            "time_range": time_range
        })
    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/set-reference-time", methods=["POST"])
def update_reference_time():
    try:
        time_str = request.json.get("reference_time")
        if not time_str:
            return jsonify({"error": "reference_time is required"}), 400
            
        reference_time = set_reference_time(time_str)
        return jsonify({
            "message": "Reference time updated successfully",
            "reference_time": reference_time.isoformat()
        })
    except Exception as e:
        logger.error(f"Error setting reference time: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)