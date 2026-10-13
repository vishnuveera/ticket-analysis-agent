# IT Ticket Analysis Agent

An AI-powered agent that analyzes IT service-management (ITSM) ticket data, identifies recurring issues, groups similar tickets into use cases, and produces analytical summaries and visualizations.

## Features

- **CSV and Excel ingestion:** Upload ticket exports and inspect workbook sheets.
- **Intelligent column detection:** Identify ticket IDs, descriptions, and resolution notes, with user confirmation.
- **LLM-powered summarization:** Generate concise technical summaries from ticket descriptions and resolution information.
- **Unsupervised clustering:** Use text embeddings, UMAP, and HDBSCAN to group similar tickets.
- **Noise identification:** Detect vague, low-information, or unclustered ticket groups.
- **Effort estimation:** Estimate effort from ticket-type mappings and allow new ticket types to be configured.
- **Natural-language analytics:** Ask questions about analyzed ticket data using the agent's SQL tools.
- **Charts and reports:** Generate Plotly chart specifications and export analysis results to Excel.
- **Session management:** Store session state, maintain conversation history, and execute longer analysis jobs in the background.
- **API integration:** Provide a FastAPI service with REST and Agent-to-Agent (A2A) integration support.

## Architecture

```text
CSV / Excel Ticket Data
          |
          v
    FastAPI / A2A
          |
          v
  LangGraph Agent
          |
     +----+------------------+
     |                       |
     v                       v
Schema and Column       Azure OpenAI
Detection              Summarization
                             |
                             v
                       Text Embeddings
                             |
                             v
                       UMAP + HDBSCAN
                             |
                             v
                  Ticket Clusters and Reports
                             |
                  +----------+----------+
                  |                     |
                  v                     v
             SQL Analytics        Charts / Excel
```

**Storage:** PostgreSQL is used for persistent session state, while SQLite supports per-session analysis data. Azure Key Vault can provide secrets, and environment variables support local configuration.

## Project Structure

| File | Purpose |
|---|---|
| `a2a_main.py` | FastAPI application and REST endpoints |
| `a2a_wrapper.py` | A2A agent card and action routing |
| `agent.py` | LangGraph orchestration and agent instructions |
| `background.py` | Background analysis tasks and status tracking |
| `config.py` | Runtime configuration and credential retrieval |
| `models.py` | Request, response, and analysis-state models |
| `session.py` | Session persistence and database rehydration |
| `tools.py` | File-analysis, SQL, and chart tools |
| `worker.py` | Summarization, embeddings, clustering, and reporting |
| `requirements.txt` | Python dependencies |
| `.env.example` | Example configuration variable names |

## Prerequisites

- A Python version supported by the pinned dependencies.
- An Azure OpenAI resource with configured chat and embedding deployments.
- A PostgreSQL database and a session table matching the configured schema.
- Azure Key Vault access or equivalent environment-variable configuration.
- Permission to process the ticket data being used.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. On Windows PowerShell, create a local configuration file:

   ```powershell
   Copy-Item .env.example .env
   ```

4. Fill in `.env` with your own Azure, API-security, encryption, and database configuration. Alternatively, supply these variables through your deployment environment.
5. Make sure the PostgreSQL schema and table specified by `PG_SCHEMA` and `PG_TABLE` exist and match the session persistence code.
6. Start the API:

   ```bash
   uvicorn a2a_main:app --host 127.0.0.1 --port 8001
   ```

7. Open `http://127.0.0.1:8001/docs` to view the FastAPI interactive documentation.

The agent discovery document is available at:

`http://127.0.0.1:8001/.well-known/ai-agent.json`

Full analysis requires working model deployments, valid credentials, and a configured database. The project has not been verified end-to-end in a clean environment merely by preparing this repository documentation.

## Example Questions

- Show the top five ticket use cases.
- Generate a chart of tickets by type.
- Which use cases have the highest estimated effort?
- Show tickets related to a particular use case.

## Security

- Never commit `.env`, API keys, database passwords, encryption keys, or certificates.
- Configure secrets through your own environment or secret manager.
- Use synthetic or anonymized ticket records for portfolio demonstrations.
- Do not upload employer, customer, personal, or otherwise confidential data.
- Before publishing the source code, confirm that you are authorized to share it and remove proprietary prompts, internal schemas, endpoints, and deployment details.

## Project Status

Portfolio preparation: documentation and configuration should be reviewed, and a clean installation and end-to-end test should be completed before claiming the application is fully reproducible.
