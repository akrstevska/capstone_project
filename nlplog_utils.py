"""
Utility functions for creating and managing synthetic log data for vector stores.
This provides test data for the RAG system even when no actual logs are available.
"""

import random
from datetime import datetime, timedelta
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Template logs for synthetic data generation
OLT_TEMPLATES = [
    "OLT FAN Open!",
    "OLT Power Supply {id} removed.",
    "OLT Temperature High: {temp}C",
    "OLT System Restart: Scheduled maintenance",
    "OLT Port {port} Link Down",
    "OLT Port {port} Link Up"
]

ONU_TEMPLATES = [
    "ONU Link LOST PON {pon} ONU {onu_id} {mac}.",
    "ONU Register PON {pon} ONU {onu_id} llid {llid} {mac}.",
    "ONU Link Event PON {pon} ONU {onu_id} {mac} Errored Symbol Period.",
    "ONU Authentication Failure PON {pon} ONU {onu_id} {mac}.",
    "ONU Signal Degraded PON {pon} ONU {onu_id} {mac}. RX Power: {power}dBm",
    "ONU Deregistration PON {pon} ONU {onu_id} {mac}. Reason: {reason}"
]

HOTSPOT_TEMPLATES = [
    "hotspot,info,debug {mac} ({ip}): trying to log in by mac",
    "hotspot,account,info,debug {mac} ({ip}): logged in",
    "hotspot,info,debug {mac} ({ip}): login failed: no more sessions are allowed for user",
    "hotspot,info,debug {mac} ({ip}): logged out after {time} idle timeout"
]

DNS_TEMPLATES = [
    "dns,error cache full, not storing",
    "dns,error cache full, not storing [ignoring repeated messages]",
    "dns,info ignoring NS without address"
]

DHCP_TEMPLATES = [
    "dhcp,warning dhcp-Users offering lease {ip} for {mac} without success",
    "dhcp,info dhcp-Users assigned {ip} to {mac}",
    "dhcp,info dhcp-Users released {ip} from {mac}"
]

# Source devices
OLT_SOURCES = [
    "TK_AZ_OLT-VE06-Garaza",
    "TK_AZ-OLT_KV02",
    "TK_AZ_OLT-VE14",
    "TK_AZ-OLT_JR07"
]

IP_SOURCES = [
    "10.252.1.3",
    "10.252.1.49",
    "10.220.16.1",
    "10.181.4.6"
]

def generate_mac():
    """Generate a random MAC address string"""
    return ':'.join([f'{random.randint(0, 255):02X}' for _ in range(6)])

def generate_ip():
    """Generate a random IP address string"""
    return '.'.join([str(random.randint(1, 254)) for _ in range(4)])

def generate_synthetic_logs(count=100):
    """
    Generate synthetic log data for testing.
    
    Args:
        count (int): Number of logs to generate
        
    Returns:
        list: List of log dictionaries
    """
    logs = []
    now = datetime.utcnow()
    
    for i in range(count):
        log_time = now - timedelta(minutes=random.randint(0, 60))
        timestamp = log_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        log_type = random.choice(["OLT", "ONU", "HOTSPOT", "DNS", "DHCP"])
        
        if log_type == "OLT":
            source = random.choice(OLT_SOURCES)
            template = random.choice(OLT_TEMPLATES)
            message = template.format(
                id=random.randint(1, 2),
                temp=random.randint(50, 85),
                port=f"{random.randint(0, 7)}/{random.randint(1, 24)}"
            )
            level = random.randint(1, 3)
        elif log_type == "ONU":
            source = random.choice(OLT_SOURCES)
            template = random.choice(ONU_TEMPLATES)
            message = template.format(
                pon=f"{random.randint(0, 7)}/{random.randint(1, 8)}",
                onu_id=random.randint(1, 64),
                mac=generate_mac(),
                llid=random.randint(1, 100),
                power=round(random.uniform(-30, -10), 1),
                reason=random.choice(["Power loss", "Admin action", "Signal degradation"])
            )
            level = random.randint(2, 4)
        elif log_type == "HOTSPOT":
            source = random.choice(IP_SOURCES)
            template = random.choice(HOTSPOT_TEMPLATES)
            message = template.format(
                mac=generate_mac(),
                ip=generate_ip(),
                time=f"{random.randint(10, 120)}m"
            )
            level = 4
        elif log_type == "DNS":
            source = random.choice(IP_SOURCES)
            message = random.choice(DNS_TEMPLATES)
            level = 3 if "error" in message else 4
        else:  # DHCP
            source = random.choice(IP_SOURCES)
            template = random.choice(DHCP_TEMPLATES)
            message = template.format(
                ip=generate_ip(),
                mac=generate_mac()
            )
            level = 3 if "warning" in message else 4
        
        # log entry
        log = {
            "timestamp": timestamp,
            "source": source,
            "message": f"{source} {message}" if log_type in ["OLT", "ONU"] else message,
            "level": level
        }
        
        logs.append(log)
    
    return logs

def load_nlplog_vectorstore(embedding_model):
    """
    Create a vector store with synthetic log data.
    
    Args:
        embedding_model: The embedding model to use
        
    Returns:
        FAISS: Vector store with synthetic log data
    """
    try:
        synthetic_logs = generate_synthetic_logs(100)
        
        documents = []
        for log in synthetic_logs:
            documents.append(
                Document(
                    page_content=log["message"],
                    metadata={
                        "source": log["source"],
                        "timestamp": log["timestamp"],
                        "level": log["level"]
                    }
                )
            )
        
        vector_store = FAISS.from_documents(documents, embedding_model)
        logger.info("Created synthetic log vector store with 100 logs")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating synthetic log vector store: {e}")
        return FAISS.from_documents([Document(page_content="Empty")], embedding_model)