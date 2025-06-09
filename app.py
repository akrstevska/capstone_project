from flask import Flask, request, jsonify
from flask_cors import CORS 

from graylog_loader import get_recent_logs, set_reference_time, get_reference_time
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
from datetime import timedelta
from sklearn.cluster import HDBSCAN
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

# def normalize_message(message):
#     """
#     Normalize log message by replacing variable parts.
#     """
#     # MAC addresses
#     normalized = re.sub(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", "XX:XX:XX:XX:XX:XX", message)
    
#     # PON identifiers
#     normalized = re.sub(r'\bPON \d+/\d+\b', 'PON X/X', normalized)
    
#     # ONU identifiers
#     normalized = re.sub(r'\bONU \d+\b', 'ONU X', normalized)
    
#     # IP addresses
#     normalized = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'X.X.X.X', normalized)
    
#     # timestamps
#     normalized = re.sub(r'\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?\b', 'YYYY-MM-DD HH:MM:SS', normalized)
    
#     return normalized

def normalize_message(message):
    """
    Normalize log message by masking only highly variable parts like MACs and IPs,
    but keeping structural and semantic components intact.
    """
    # MAC addresses
    message = re.sub(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", "<MAC>", message)
    
    # IP addresses
    message = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", message)
    
    # LLID (logical link ID) if exists
    message = re.sub(r"\bllid \d+\b", "llid <ID>", message)
    
    # Optional: remove specific ONU serials (but not ONU ID numbers like 'ONU 1')
    message = re.sub(r"\b[A-Z0-9]{4}\.[A-Z0-9]{4}\b", "<SERIAL>", message)
    
    return message.strip()



def select_diverse_logs(logs, max_per_type=5, max_total=40):
    """
    Selects diverse logs, prioritizing newer entries first.
    Groups logs by normalized message and limits how many per type.
    """
    def parse_time(log):
        try:
            return datetime.fromisoformat(log.get("timestamp").replace("Z", "+00:00"))
        except:
            return datetime.min 

    logs = sorted(logs, key=parse_time, reverse=True)

    grouped = defaultdict(list)
    for log in logs:
        norm_msg = normalize_message(log["message"])
        grouped[norm_msg].append(log)

    selected = []
    for group in grouped.values():
        selected.extend(group[:max_per_type])
        if len(selected) >= max_total:
            break

    return selected[:max_total]

def should_filter_logs(user_q: str) -> bool:
    broad_keywords = ['ip', 'issue', 'issues', 'error', 'problem', 'trend', 'happened', 'pattern', 'failures', 'report', 'deregistration']

    if any(word in user_q.lower() for word in broad_keywords):
        return False
    return True 

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
            max_logs=10000
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

@app.route("/log-storms", methods=["GET"])
def log_storms():
    try:
        logs = get_recent_logs(max_logs=10000)
        if not logs:
            return jsonify({"error": "No logs found."})

        buckets = defaultdict(lambda: defaultdict(int))

        for log in logs:
            level = log.get("level", "6")
            
            if str(level) not in ["-1", "0", "1", "2", "3"]:
                continue
            
            msg = normalize_message(log.get("message", ""))
            if not msg or msg.strip() == "":
                continue
            
            ts = log.get("timestamp")
            if not ts or len(str(ts)) < 16:
                continue
                
            minute_bucket = str(ts)[:16]  # e.g., "2025-06-08T12:43"
            buckets[msg][minute_bucket] += 1

        result = {}
        for msg, times in buckets.items():
            if max(times.values()) >= 2:
                result[msg] = dict(sorted(times.items()))

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in log_storms endpoint: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/rule-clusters", methods=["GET"])
def rule_clusters():
    try:
        logs = get_recent_logs(max_logs=10000)
        if not logs:
            return jsonify({"error": "No logs found."})

        rule_groups = defaultdict(list)
        for log in logs:
            msg = log.get("message", "")
            norm = normalize_message(msg)

            if "DHCP" in norm:
                rule_groups["DHCP Issues"].append(log)
            elif "ONU Link Event" in norm:
                rule_groups["ONU Link Drops"].append(log)
            elif "fetch" in norm:
                rule_groups["Fetch Events"].append(log)
            elif "hotspot" in norm:
                rule_groups["Hotspot Activity"].append(log)
            else:
                rule_groups["Miscellaneous"].append(log)

        summarized = {}
        for rule, items in rule_groups.items():
            summarized[rule] = [
                {
                    "message": normalize_message(item.get("message", "")),
                    "source": item.get("source", "Unknown"),
                    "timestamp": item.get("timestamp"),
                    "level": item.get("level")
                }
                for item in items[:30]  
            ]

        return jsonify(summarized)

    except Exception as e:
        logger.error(f"Error in rule_clusters endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# @app.route("/ask", methods=["POST"])
# def ask():
#     try:
#         user_q = request.json.get("question")
#         style = request.json.get("style", "summary")
#         skip_llm = request.json.get("skip_llm", False)

#         reference_time = request.json.get("reference_time")
#         if reference_time:
#             set_reference_time(reference_time)
        
#         filters = lp.extract_query_filters(user_q)
#         if "target_ip" in filters:
#             filters["source"] = filters.pop("target_ip")
#         time_filters = {k: v for k, v in filters.items() 
#                        if k in ['minutes_ago', 'hours_ago', 'today', 'yesterday']}
#         logger.info(f"Extracted time filters: {time_filters}")
        
#         logs = get_recent_logs(
#             max_logs=20000,
#             minutes=time_filters.get('minutes_ago'),
#             hours=time_filters.get('hours_ago'),
#             source_ip=filters.get("source")
            
#         )
        
#         if not logs:
#             return jsonify({"answer": "No logs found for the specified time period."})
#         if filters:
#             filtered_logs = []
#             for log in logs:
#                 enriched = lp.enrich_log_metadata(log)
#                 match = True
#                 for key in ["onu_id", "pon", "device_id"]:
#                     if key in filters and str(enriched.get(key)) != str(filters[key]):
#                         match = False
#                         break
#                 if match:
#                     filtered_logs.append(enriched)
#             logs = filtered_logs
            
#         k_logs = 20  
#         if style == "detailed":
#             k_logs = 60
#         elif style == "critical":
#             k_logs = 80
#         elif style == "report":
#             k_logs = 100
#         logger.info(f"Filtered logs count before diversity selection: {len(logs)}")
   
#         if should_filter_logs(user_q):
#             logs = select_diverse_logs(logs, max_per_type=3, max_total=k_logs)

#         if style == "summary":
#             system_prompt = short_summary_prompt(user_q)
#         elif style == "detailed":
#             system_prompt = detailed_analysis_prompt(user_q)
#         elif style == "critical":
#             system_prompt = critical_events_prompt(user_q)
#         elif style == "report":
#             system_prompt = report_generator_prompt(user_q)
#         else:
#             system_prompt = short_summary_prompt(user_q)
            
#         vector_store = lp.create_enhanced_vector_store(logs, embedding_model)
        
#         llm = OllamaLLM(model="llama3")
        
#         if filters:
#             filter_function = lp.create_filter_function(filters)
#             retriever = vector_store.as_retriever(
#                 search_kwargs={
#                     "k": k_logs,
#                     "filter": filter_function
#                 }
#             )
#         else:
#             retriever = vector_store.as_retriever(search_kwargs={"k": k_logs})
        
#         qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        
#         answer = None
#         if not skip_llm:
#             try:
#                 response = qa.invoke(system_prompt)
#                 if isinstance(response, dict) and "result" in response:
#                     answer = response["result"]
#                 else:
#                     answer = qa.run(system_prompt)
#                     logger.info("Used legacy run() method as fallback")
#             except Exception as e:
#                 logger.warning(f"Error with invoke method: {e}")
#                 answer = qa.run(system_prompt) 
#                 logger.info("Used legacy run() method as fallback")

#         relevant_logs = []
#         try:
           
#             relevant_docs = vector_store.similarity_search(user_q, k=k_logs)
            
#             for doc in relevant_docs:
#                 try:
#                     metadata = {}
#                     content = ""
                    
#                     if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
#                         metadata = doc.metadata
#                         content = doc.page_content
#                     elif isinstance(doc, dict):
#                         metadata = doc.get('metadata', {})
#                         content = doc.get('page_content', str(doc))
#                     else:
#                         content = str(doc)
                    
#                     source = metadata.get('source', 'Unknown')
#                     level = metadata.get('level', 'Unknown')
#                     timestamp = metadata.get('timestamp', 'Unknown')
#                     device_type = metadata.get('device_type', None)
#                     device_id = metadata.get('device_id', 'Unknown')
#                     onu_id = metadata.get('onu_id', None)
#                     pon = metadata.get('pon', None)
                    
#                     metadata_parts = [
#                         f"Source: {source}",
#                         f"Level: {level}",
#                         f"Time: {timestamp}"
#                     ]
                    
#                     if device_type == "OLT" and device_id:
#                         metadata_parts.append(f"Device: {device_id}")
#                     elif device_type == "ONU":
#                         if onu_id and pon:
#                             metadata_parts.append(f"ONU {onu_id} on PON {pon}")
#                         elif onu_id:
#                             metadata_parts.append(f"ONU {onu_id}")
                    
#                     metadata_str = " | ".join(metadata_parts)
#                     formatted_log = f"[{metadata_str}] {content}"
#                     relevant_logs.append(formatted_log)
                    
#                 except Exception as e:
#                     logger.warning(f"Error formatting individual log: {e}")
#                     relevant_logs.append(f"[Raw Log] {str(doc)}")
                    
#         except Exception as e:
#             logger.error(f"Error retrieving or processing relevant logs: {e}")
#             relevant_logs = ["Error retrieving supporting logs"]

#         # result
#         return jsonify({
#             "answer": answer if answer else "LLM skipped",
#             "supporting_logs": relevant_logs,
#             "metadata": {
#                 "retrieved_count": len(relevant_logs),
#                 "applied_filters": filters,
#                 "total_logs_searched": len(logs),
#                 "time_filters_applied": time_filters
#             }
#         })
#     except Exception as e:
#         logger.error(f"Error in ask endpoint: {e}")
#         return jsonify({"error": str(e)}), 500
@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_q = request.json.get("question")
        style = request.json.get("style", "summary")
        skip_llm = request.json.get("skip_llm", False)

        reference_time = request.json.get("reference_time")
        if reference_time:
            set_reference_time(reference_time)

        filters = lp.extract_query_filters(user_q)
        if "target_ip" in filters:
            filters["source"] = filters.pop("target_ip")

        time_filters = {k: v for k, v in filters.items() 
                        if k in ['minutes_ago', 'hours_ago', 'today', 'yesterday']}
        logger.info(f"Extracted time filters: {time_filters}")

        logs = get_recent_logs(
            max_logs=20000,
            minutes=time_filters.get('minutes_ago'),
            hours=time_filters.get('hours_ago'),
            source_ip=filters.get("source")
        )

        if not logs:
            return jsonify({"answer": "No logs found for the specified time period."})

        # Enrich and filter logs
        if filters:
            filtered_logs = []
            for log in logs:
                enriched = lp.enrich_log_metadata(log)
                match = True
                for key in ["onu_id", "pon", "device_id"]:
                    if key in filters and str(enriched.get(key)) != str(filters[key]):
                        match = False
                        break
                if match:
                    filtered_logs.append(enriched)
            logs = filtered_logs

        # Style-specific log count
        k_logs = 20
        if style == "detailed":
            k_logs = 60
        elif style == "critical":
            k_logs = 80
        elif style == "report":
            k_logs = 100

        logger.info(f"Filtered logs count before diversity selection: {len(logs)}")

        if should_filter_logs(user_q):
            logs = select_diverse_logs(logs, max_per_type=3, max_total=k_logs)

        # Create vector store
        vector_store = lp.create_enhanced_vector_store(logs, embedding_model)
        llm = OllamaLLM(model="llama3")

        answer = None
        relevant_logs = []

        if not skip_llm:
            try:
                # Get top relevant docs
                relevant_docs = vector_store.similarity_search(user_q, k=k_logs)

                # Format retrieved logs for the prompt and output
                context_logs = []
                for doc in relevant_docs:
                    meta = doc.metadata
                    log_line = f"[{meta.get('timestamp', 'Unknown')}] [{meta.get('source', 'Unknown')}] [Level {meta.get('level', 'Unknown')}] {doc.page_content}"
                    context_logs.append(log_line)
                    # also store for output
                    relevant_logs.append(log_line)

                logs_context = "\n".join(context_logs)

                # Select prompt template
                if style == "summary":
                    base_prompt = short_summary_prompt(user_q)
                elif style == "detailed":
                    base_prompt = detailed_analysis_prompt(user_q)
                elif style == "critical":
                    base_prompt = critical_events_prompt(user_q)
                elif style == "report":
                    base_prompt = report_generator_prompt(user_q)
                else:
                    base_prompt = short_summary_prompt(user_q)

                # Construct full prompt
                full_prompt = f"""{base_prompt}

AVAILABLE LOG ENTRIES:
{logs_context}

Based on the log entries above, provide your analysis:"""

                # Run LLM
                answer = llm.invoke(full_prompt)

            except Exception as e:
                logger.error(f"Error during LLM log analysis: {e}")
                answer = "Error processing logs with LLM"

        # Return final response
        return jsonify({
            "answer": answer if answer else "LLM skipped",
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

    
        logs = get_recent_logs(
            max_logs=None,
            hours=24,
            minutes=None
        )

        if not logs:
            return jsonify({"error": "No logs found."})

        total_logs = len(logs)
        level_counts = Counter([log.get("level", 6) for log in logs])
        source_counts = Counter([log.get("source", "Unknown") for log in logs])

        devices = set()
        for log in logs:
            device_info = lp.extract_device_info(log.get("message", ""))
            if device_info["device_id"]:
                devices.add(device_info["device_id"])

        ref_time = get_reference_time()
        split_time = ref_time - timedelta(hours=12)

        recent_12h = [log for log in logs if log.get("datetime") >= split_time]
        prev_12h = [log for log in logs if log.get("datetime") < split_time]

        def compute_stats(log_subset):
            count = len(log_subset)
            error_logs = [log for log in log_subset if 0 <= log.get("level", 6) <= 3]
            error_rate = (len(error_logs) / count) * 100 if count else 0
            return count, error_rate

        recent_count, recent_error = compute_stats(recent_12h)
        prev_count, prev_error = compute_stats(prev_12h)

        log_volume_change = ((recent_count - prev_count) / prev_count) * 100 if prev_count else 0
        error_rate_change = (recent_error - prev_error)

        sorted_logs = sorted(logs, key=lambda l: l.get("datetime"), reverse=True)
        recent_logs = [
            {
                "level": log.get("level"),
                "message": log.get("message"),
                "timestamp": log.get("timestamp")
            }
            for log in sorted_logs[:10]
        ]

        return jsonify({
            "total_logs": total_logs,
            "logs_by_level": {str(k): v for k, v in level_counts.items()},
            "logs_by_source": dict(source_counts.most_common(10)),
            "unique_devices": len(devices),
            "sample_devices": list(devices)[:10] if devices else [],
            "time_range": "last 24 hours",

            "dashboard_summary": {
                "error_rate_percent": round((level_counts[0] + level_counts[1] + level_counts[2] + level_counts[3]) / total_logs * 100, 2),
                "log_volume_change_percent": round(log_volume_change, 1),
                "error_rate_change_percent": round(error_rate_change, 1),
                "total_logs_last_12h": recent_count,
                "total_logs_prev_12h": prev_count,
                "error_rate_last_12h": round(recent_error, 1),
                "error_rate_prev_12h": round(prev_error, 1),
                "unique_devices": len(devices)
            },

            "recent_logs": recent_logs
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