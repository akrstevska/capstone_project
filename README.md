# Log Query Examples and Project Structure

## 🔁 API Usage

### `/ask` Endpoint
This route expects a JSON payload with the following fields:

- `"question"`: A natural language prompt (e.g., `"What happened to ONU 1 on PON 0/7 in the last hour?"`)
- `"style"`: A string defining the response type. One of:
  - `"summary"`
  - `"detailed"`
  - `"critical"`
  - `"report"`

---

## Sample Queries

### `Detailed` style
**Query:** Any issues from device TK_AZ-OLT_KV02?

---

### `Critical` style
**Query:** Filter and summarize critical events from the last 30 minutes.

---

### `Summary` style
**Query:** Summarize logs involving ONU deregistration.

---

### `Detailed` style
**Query:** What happened to ONU 1 on PON 0/7 in the last hour?

---

### `Report` style
**Query:** Generate a daily log report in bullet points.

---

## Project Structure

- `app.py`: Flask backend handling query endpoints like `/ask`, `/stats`, `/clustered-logs`
- `graylog_loader.py`: Handles loading and filtering logs from CSV (or future Graylog integration)
- `log_processing.py`: Extracts and enriches metadata from raw logs (e.g., ONU IDs, PONs)
- `prompt_templates.py`: Templates used to control the LLM behavior (summary, detailed, etc.)
- `nlplog_utils.py`: Synthetic log generation and FAISS vector store setup (not used)
- `logs-24h.csv`: Main input file containing system logs for the past 24 hours

## Notes
- Time filtering is based on a fixed reference time: `2025-05-16 12:55:00` (Europe/Belgrade)
