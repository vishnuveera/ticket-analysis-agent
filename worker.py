import os
import pandas as pd
import numpy as np
import hdbscan
import umap
import random
import asyncio
import logging
from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Local Imports
from config import (
    AZURE_DEPLOYMENT, AZURE_MINI_DEPLOYMENT, AZURE_EMBEDDING_DEPLOYMENT,
    AZURE_API_VERSION, AZURE_ENDPOINT, logger, get_azure_api_key  
)
from models import SheetSchema, UsecaseInfo, NoiseAudit

class TicketAnalysisAgent:
    
    def __init__(self):
        logger.info("Initializing TicketAnalysisAgent...")
        self.llm = AzureChatOpenAI(
            deployment_name=AZURE_DEPLOYMENT, 
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=get_azure_api_key(),
            temperature=0.1, 
            max_tokens=10000,
            model_kwargs={} 
        )

        
        self.llm_mini = AzureChatOpenAI(
            deployment_name=AZURE_MINI_DEPLOYMENT, 
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=get_azure_api_key(),
            temperature=0.1, 
            max_tokens=5000, 
            model_kwargs={} 
        )

        self.embedder = AzureOpenAIEmbeddings(
            api_version="2024-12-01-preview",
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=get_azure_api_key(),
            model_kwargs={}
        )
        logger.info("TicketAnalysisAgent initialized successfully.")
        
        self.TICKET_TYPE_MAP = {"INC": "Incident", "SR": "Service Request","HC":"Health Check"}
        self.EFFORT_MAP = {"Incident": 2, "Service Request": 4, "Health Check":4}
        self.MIN_CLUSTER_SIZE = 10
        self.BATCH_SIZE = 100

    async def classify_unknown_ticket_types(
        self, 
        df: pd.DataFrame, 
        id_col: str, 
        summary_col: str = None, 
        file_path: str = None
    ) -> Dict[str, Any]:
        """
        Identifies ticket ID prefixes not in TICKET_TYPE_MAP and uses LLM to classify them.
        Returns a dict with info about unknown prefixes and their suggested classifications.
        """
        # Extract prefixes
        prefixes = df[id_col].astype(str).str.extract(r'^([A-Za-z]+)')[0].str.upper()
        
        all_prefixes = prefixes.dropna().unique()
        known_prefixes = set(self.TICKET_TYPE_MAP.keys())
        unknown_prefixes = [p for p in all_prefixes if p not in known_prefixes]
        
        # Check for numeric-only IDs
        has_numeric = prefixes.isna().any()
        numeric_count = int(prefixes.isna().sum()) if has_numeric else 0
        
        if not unknown_prefixes and not has_numeric:
            return {}  # All ticket types are known
        
        # Build context for LLM
        file_name = os.path.basename(file_path) if file_path else "Unknown"
        context_parts = []
        
        for prefix in unknown_prefixes:
            mask = prefixes == prefix
            count = int(mask.sum())
            sample_ids = df.loc[mask, id_col].head(10).tolist()
            sample_summaries = []
            if summary_col and summary_col in df.columns:
                sample_summaries = df.loc[mask, summary_col].dropna().head(5).tolist()
            
            context_parts.append({
                "prefix": prefix,
                "count": count,
                "sample_ids": sample_ids,
                "sample_summaries": sample_summaries
            })
        
        if has_numeric:
            numeric_mask = prefixes.isna()
            numeric_sample_ids = df.loc[numeric_mask, id_col].head(10).tolist()
            numeric_summaries = []
            if summary_col and summary_col in df.columns:
                numeric_summaries = df.loc[numeric_mask, summary_col].dropna().head(5).tolist()
            
            context_parts.append({
                "prefix": "NUMERIC",
                "count": numeric_count,
                "sample_ids": numeric_sample_ids,
                "sample_summaries": numeric_summaries
            })
        
        # Call LLM to classify
        CLASSIFY_PROMPT = """You are an ITSM (IT Service Management) expert. You will be given information about unknown ticket ID prefixes found in a ticket data file.

For each unknown prefix, classify it into a meaningful ticket type category name.

Common ITSM ticket types include:
- Change Request (prefixes: CRQ, CHG, CR)
- Task (prefixes: TASK, TSK)
- Problem (prefixes: PRB, PBI)
- Request (prefixes: REQ, RQ)
- Catalog Task (prefixes: SCTASK, RITM)
- Knowledge Article (prefixes: KB)
- General Ticket (for numeric-only IDs)

Use the sample ticket IDs and summaries provided to help determine the correct type.

Respond with ONLY a JSON object mapping each prefix to its suggested type name.
Example: {{"CRQ": "Change Request", "TASK": "Task", "NUMERIC": "General Ticket"}}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", CLASSIFY_PROMPT),
            ("user", "File name: {file_name}\n\nUnknown prefixes:\n{context}")
        ])
        
        chain = prompt | self.llm_mini | JsonOutputParser()
        
        try:
            context_str = "\n".join([
                f"Prefix: {p['prefix']}, Count: {p['count']}, "
                f"Sample IDs: {p['sample_ids'][:10]}, "
                f"Sample Summaries: {p['sample_summaries'][:5]}"
                for p in context_parts
            ])
            
            llm_result = await chain.ainvoke({"file_name": file_name, "context": context_str})
            logger.info(f"LLM classified unknown ticket types: {llm_result}")
            
            # Build output with counts
            output = {}
            for part in context_parts:
                prefix = part["prefix"]
                suggested_name = llm_result.get(prefix, prefix)
                output[prefix] = {
                    "suggested_type_name": suggested_name,
                    "count": part["count"],
                    "sample_ids": part["sample_ids"][:5]
                }
            
            return output
            
        except Exception as e:
            logger.error(f"Failed to classify unknown ticket types: {e}")
            # Fallback: use prefix as type name
            output = {}
            for part in context_parts:
                prefix = part["prefix"]
                output[prefix] = {
                    "suggested_type_name": prefix if prefix != "NUMERIC" else "General Ticket",
                    "count": part["count"],
                    "sample_ids": part["sample_ids"][:5]
                }
            return output

    def load_data(self, file_path: str) -> Dict[str, pd.DataFrame]:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            encodings_to_try = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully read CSV with encoding: {encoding}")
                    return {"Sheet1": df}
                except UnicodeDecodeError:
                    continue 
                except Exception as e:
                    raise ValueError(f"Failed to read CSV: {e}")
            
            raise ValueError("Could not read CSV file. Unknown encoding.")

        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, sheet_name=None)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def get_sheet_preview(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "This sheet is empty."
        headers = list(df.columns)
        sample_data = df.head(5).to_string(index=False)
        return f"Headers: {headers}\n\nSample Data:\n{sample_data}"

    async def find_schema(self, sheet_name: str, df: pd.DataFrame) -> SheetSchema:
        """
        Uses LLM to find the correct columns for a *single* sheet.
        """
        logger.info(f"Analyzing schema for sheet: '{sheet_name}'")
        preview = self.get_sheet_preview(df)
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert data analyst. Identify the `id_column`, `summary_column` (ticket description), and optionally the `resolution_column` (technician notes/solution) for the sheet. Respond *only* with the JSON structure."),
            ("user", f"Here is the preview for sheet '{sheet_name}':\n\n{preview}")
        ])
        chain = prompt | self.llm | JsonOutputParser(pydantic_object=SheetSchema)
        
        logger.info("Calling LLM to extract schema...")
        
        try:
            partial_schema = await chain.ainvoke({})
            if 'sheet_name' not in partial_schema:
                partial_schema['sheet_name'] = sheet_name
        except Exception as e:
            logger.error(f"Failed to parse LLM schema output: {e}")
            if isinstance(partial_schema, dict):
                 partial_schema['sheet_name'] = sheet_name
            else:
                 raise e
        
        logger.info("LLM extraction successful.")
        return SheetSchema(**partial_schema)
    
    async def generate_session_title(self, ticket_count: int, executive_summary: str) -> str:
        """Generates a short 3-5 word title based on the analysis summary."""
        
        context_str = f"Total Tickets: {ticket_count} | Key Findings: {executive_summary[:500]}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional analyst. Generate a distinct, business-focused session title (3-6 words) based on the IT Ticket analysis context. Focus on the main technology or failure themes (e.g., 'VPN & Outlook Issues', 'SAP Access Failures'). Do not use quotes."),
            ("user", f"Context: {context_str}")
        ])
        
        # Use mini model for speed
        chain = prompt | self.llm_mini 
        try:
            res = await chain.ainvoke({})
            title = res.content.strip().replace('"', '').replace("'", "")
            return title
        except Exception as e:
            logger.warning(f"Title generation failed: {e}")
            return "Ticket Analysis Session"

    async def _get_one_rich_summary(self, ticket_text: str) -> str:

        SYSTEM_PROMPT_TICKET = """
You are an expert IT technical analyst. Your goal is to process an IT ticket into a clean, technical summary.

You will receive text containing the 'Summary' and 'Resolution Note'. Both are given for your information,
both or any one of them will contain the necessary information

From this text, you should identify:
1.  **Technology:** The core technology or system involved (e.g., 'Cisco', 'M365', 'Azure').
2.  **Component:** The specific part affected (e.g., 'Access Point', 'Outlook', 'Load Balancer').
3.  **Action/Resolution:** The primary resolution step if there is any (e.g., 'Reset', 'Profile rebuilt', 'State verified').

**PHASE: SUMMARIZATION**
If the ticket is valid, write a SINGLE, concise sentence following this format:
"[Technology/System] [Component] [Specific Issue]."

**RULES FOR SUMMARIES:**
- **REMOVE PROCEDURAL TEXT:** Do NOT mention "Duplicate incident", "Ticket closed", "User contacted", "SLA breached".
- **REMOVE FILLER:** Do NOT start with "The issue was...", "Resolution involved...".
- **DO NOT** include any non-technical details like server names (e.g., 'gbr-r-02119-01'), hostnames, serial numbers, incident numbers, or user names.
- **DO** be specific. 'M365 Outlook profile' is better than 'User profile'.
- **EXPAND ACRONYMS:** 'AP' -> 'Access Point', 'LB' -> 'Load Balancer'.
- **FOCUS ON THE FIX:** If the problem is vague but the resolution is "Replaced hard drive", assume the issue was "Hard Drive Failure".

**EXAMPLES:**

Input: Azure Resource 'nprod-pharmacy-northeurope-ilb-01' generated an Alert | LB in available state
Output: Azure Virtual Machine Load Balancer state alert.

Input: Outlook is crashing | User profile was corrupted. Rebuilt profile and cleared cache.
Output: M365 Outlook client crashing due to corrupted User profiles.

Input:Summary: AP down | AP was reset.
Output: Cisco Access Point was unresponsive.

Input: The high CPU usage on the SQL server | Monitoring alert, found runaway query and killed it.
Output: Microsoft SQL Server performance issue caused by high CPU usage due to runaway query.

Input: "Azure Resource 'nprod-pharmacy' Alert | planned maintenance completed"
Output: Azure Virtual Machine Alert resolved by verifying planned maintenance completion.

Input: "LMD10 error CommVault Job Backups | duplicate of INC342"
Output: CommVault Backup Job failed with error status.
(Note: We ignored the "duplicate" part and focused on the error).
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_TICKET),
            ("user", "{ticket_text}")
        ])
        
        chain = prompt | self.llm_mini
        try:
            msg = await chain.ainvoke({"ticket_text": ticket_text})
            return msg.content
        except Exception as e:
            logger.error(f"Failed to summarize: {e}")
            return ticket_text 

    async def generate_rich_summaries(
        self, 
        summaries: List[str], 
        logs: List[str] = None
    ) -> List[str]:
        
        status_msg = f"Generating rich summaries for {len(summaries)} tickets..."
        if logs is not None: logs.append(status_msg)
        logger.info(status_msg)
        
        generated_summaries = []
        
        for i in range(0, len(summaries), self.BATCH_SIZE):
            batch = [self._get_one_rich_summary(s) for s in summaries[i:i + self.BATCH_SIZE]]
            
            batch_msg = f"  - Processing summary batch {i//self.BATCH_SIZE + 1}..."
            if logs is not None: logs.append(batch_msg)
            logger.info(batch_msg)
            
            results = await asyncio.gather(*batch, return_exceptions=True)
            generated_summaries.extend(
                str(r) if not isinstance(r, Exception) else "Error: Summary failed" 
                for r in results
            )
        
        if logs is not None: logs.append("✅ Rich summaries complete.")
        logger.info("Rich summaries complete.")
        return generated_summaries

    def cluster_data(
        self, 
        df: pd.DataFrame, 
        embeddings: np.ndarray, 
        logs: List[str] = None
    ) -> pd.DataFrame:
        
        if logs is not None: logs.append("Clustering (UMAP + HDBSCAN) per ticket type...")
        logger.info("Starting Per-Type Clustering with Unique IDs...")
        
        # Initialize default usecase_id to -2
        df['usecase_id'] = -2 
        
        # Tracks the highest ID used so far to ensure uniqueness across types
        global_cluster_offset = 0
        
        for ticket_type in df['ticket_type'].dropna().unique():
            
            type_mask = (df['ticket_type'] == ticket_type)
            if not type_mask.any(): 
                continue
            
            type_embeddings = embeddings[type_mask]
            num_tickets = len(type_embeddings)
            
            # --- SCALING LOGIC: Dynamic Cluster Sizing ---
            # Ensures granularity for 600 tickets, while preventing 10,000 unusable micro-clusters at 60k tickets
            if num_tickets <= 1000:
                dynamic_min_cluster_size = 3
            elif num_tickets <= 10000:
                dynamic_min_cluster_size = 10
            elif num_tickets <= 30000:
                dynamic_min_cluster_size = 30
            else:
                dynamic_min_cluster_size = 30

            if num_tickets < dynamic_min_cluster_size:
                logger.info(f"Skipping {ticket_type}: Count {num_tickets} < Min {dynamic_min_cluster_size}")
                df.loc[type_mask, 'usecase_id'] = -1 # Mark as noise/unclustered
                continue

            try:
                # UMAP Tweaks: Dynamically scale dimensions to prevent sparse data stretching on small files
                n_neighbors = min(15, num_tickets - 1)
                n_components = min(15, max(2, num_tickets - 2)) if num_tickets < 50 else 50
                
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors, 
                    n_components=n_components, 
                    metric='cosine', 
                    random_state=42, 
                    low_memory=True
                )
                reduced_embeddings = reducer.fit_transform(type_embeddings)

                # --- HDBSCAN Tweaks ---
                # 'leaf' forces fine-grained clusters. min_samples prevents smaller groups from being discarded.
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=dynamic_min_cluster_size,
                    min_samples=max(2, dynamic_min_cluster_size // 2),
                    metric='euclidean',
                    gen_min_span_tree=True,
                    cluster_selection_method='leaf' 
                )
                raw_labels = clusterer.fit_predict(reduced_embeddings)
                
                # --- UNIQUE ID LOGIC ---
                shifted_labels = raw_labels.copy()
                valid_mask = (raw_labels >= 0)
                
                if valid_mask.any():
                    shifted_labels[valid_mask] += global_cluster_offset
                    num_new_clusters = raw_labels.max() + 1
                    global_cluster_offset += num_new_clusters

                df.loc[type_mask, 'usecase_id'] = shifted_labels
                
                n_clusters_found = len(set(shifted_labels)) - (1 if -1 in shifted_labels else 0)
                logger.info(f"Type: {ticket_type} | Found {n_clusters_found} clusters. Next ID starts at: {global_cluster_offset}")

            except Exception as e:
                logger.error(f"Clustering failed for {ticket_type}: {e}")
                df.loc[type_mask, 'usecase_id'] = -1

        if logs is not None: logs.append("✅ Usecase generation complete.")
        return df

    async def generate_cluster_names(
        self, 
        df_with_clusters: pd.DataFrame, 
        logs: List[str] = None,
        duration_months: int = 1,
        effort_map: Dict[str, int] = None
    ) -> pd.DataFrame:
        
        if logs is not None: logs.append("Naming Usecases ...")
        logger.info("Naming clusters...")
        
        unique_clusters = df_with_clusters[df_with_clusters['usecase_id'] >= 0][['ticket_type', 'usecase_id']].drop_duplicates()
        
        SYSTEM_PROMPT_CLUSTER = """
        You are a senior IT Operations Manager. You will be given a list of individual ticket summaries that have been algorithmically grouped 
into a cluster.

Your goal is to synthesize these and to output a JSON object with two fields:
        1. `usecase_name`: A concise title (5-10 words).
        2. `technical_description`: A dense, technical summary optimized for vector database matching.

**concise title** that describes the common **problem theme** of the entire cluster.

**INSTRUCTIONS FOR `usecase_name`:**
This title should clearly state:
1.  **The core technology** or system involved.
2.  **The common problem** or component being affected.

- Your response MUST be a single, concise title (5-10 words).
- Do not use markdown or newlines.
- Do not say "This cluster represents...". Just state the theme.

**Example Input (list of summaries):**
- M365 Outlook client crashing resolved by rebuilding the user profile...
- Microsoft 365 Outlook issue resolved by profile recreation...
- O365 Outlook client resolved by clearing the cache and rebuilding...

**Example Output (your title):**
M365 Outlook Client Profile Corruption

**INSTRUCTIONS FOR `technical_description`:**
        - Write a single, information-dense sentence summarizing the technical issue.
        - **DO NOT** use conversational filler (e.g., "The user reported...", "This issue is...").
        - **DO NOT** use repetitive labels (e.g., "Issue:", "Cause:", "Context:").
        - **DO** include specific error codes, component names, and symptom descriptors.
        
        **Examples:**
        - *Bad:* "Users are facing crashes in Outlook which is caused by bad profiles."
        - *Bad:* "Issue: Crash; Component: Outlook; Cause: OST."
        - *Good:* "Microsoft Outlook client crashing on startup due to corrupted local OST configuration and profile integrity failures."
        - *Good:* "VPN remote access connection failure returning Error 800 due to Radius server timeout."

**SCENARIO EXAMPLES:**

* **Bad Title:** "Azure Virtual Machine Alerts and Resolutions"
    * **Reason:** Contains "and Resolutions".
    * **Correct Title:** "Azure Virtual Machine Performance Alerts"

* **Bad Title:** "CommVault Backup Job Errors and Duplicate Incidents"
    * **Reason:** Contains "Duplicate Incidents".
    * **Correct Title:** "CommVault Backup Job Failures"

* **Bad Title:** "Outlook Profile Rebuilds"
    * **Reason:** Describes the *fix*, not the *problem*.
    * **Correct Title:** "Outlook Client Profile Corruption"
        
Respond *only* with the JSON structure: {{"usecase_name": "...", "technical_description": "..."}}
"""
        
        parser = JsonOutputParser(pydantic_object=UsecaseInfo)
        prompt = ChatPromptTemplate.from_messages([
             ("system", SYSTEM_PROMPT_CLUSTER),
             ("user", "List of summaries:\n{sample_summaries}")
        ])
        chain = prompt | self.llm_mini | parser

        tasks = []
        task_data = [] 

        for _, row in unique_clusters.iterrows():
            ttype, cid = row['ticket_type'], row['usecase_id']
            
            # Get summaries for this specific cluster
            cluster_summaries = df_with_clusters[
                (df_with_clusters['ticket_type'] == ttype) & (df_with_clusters['usecase_id'] == cid)
            ]['rich_summary'].tolist()
            
            ticket_count = len(cluster_summaries)
            if ticket_count == 0:
                continue
            
            # --- SAMPLING LOGIC ---
            sample_size = 30
            if len(cluster_summaries) > sample_size:
                sample = random.sample(cluster_summaries, sample_size)
            else:
                sample = cluster_summaries
            
            tasks.append(chain.ainvoke({"sample_summaries": "\n- ".join(sample)}))
            task_data.append((ttype, cid, ticket_count))

        if not tasks:
            return pd.DataFrame()

        # Cap the batch size at 50 to strictly enforce OS file descriptor limits 
        # and Azure OpenAI rate limits, even if self.BATCH_SIZE is set higher.
        safe_batch_size = min(50, getattr(self, 'BATCH_SIZE', 100))
        total_tasks = len(tasks)

        if logs is not None: 
            logs.append(f"  - Identifying Names for {total_tasks} Usecases in batches of {safe_batch_size}...")
        
        results = []
        
        # Iterate through the tasks array in chunks
        for i in range(0, total_tasks, safe_batch_size):
            batch = tasks[i:i + safe_batch_size]
            
            batch_msg = f"  - Processing cluster naming batch {i // safe_batch_size + 1}/{(total_tasks + safe_batch_size - 1) // safe_batch_size}..."
            logger.info(batch_msg)
            
            # Await the current chunk and append to the main results list
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)
        
        # --- Results Processing ---
        cluster_report_data = []
        for i, res in enumerate(results):
            ttype, cid, ticket_count = task_data[i]

            # CALCULATE PER MONTH METRICS
            # Round to nearest whole number and convert to integer for tickets
            tickets_per_month = int(round(ticket_count / duration_months)) 
            # Use run-specific effort map if provided, else fall back to default
            active_effort_map = effort_map if effort_map else self.EFFORT_MAP
            efforts_per_month = tickets_per_month*active_effort_map.get(ttype,0)
            
            # Calculate effort
            effort = ticket_count * active_effort_map.get(ttype, 0)
            
            if isinstance(res, Exception):
                logger.warning(f"Could not name cluster {cid}.", exc_info=res)
                cluster_name = f"Cluster {cid} (Naming Failed)"
                tech_desc = "Analysis Failed" 
            else:
                parsed = UsecaseInfo(**res)
                cluster_name = parsed.usecase_name
                tech_desc = parsed.technical_description

            cluster_report_data.append({
                "usecase_name": cluster_name,
                "technical_description": tech_desc, 
                "ticket_type": ttype,
                "total_tickets": ticket_count,              
                "total_effort_hours": effort,  
                "tickets_per_month": tickets_per_month,
                "efforts_required_per_month_in_hours": efforts_per_month, 
                "usecase_id": cid,
            })
            
        return pd.DataFrame(cluster_report_data)

    @staticmethod
    def _normalize_cluster_name(name: str) -> str:
        """Normalize cluster name for comparison: lowercase + basic plural stripping."""
        words = str(name).lower().strip().split()
        normalized = []
        for w in words:
            # Strip trailing 's' for singular/plural normalization
            # Protect words ending in 'ss' (access, process) or 'us' (status)
            if len(w) > 3 and w.endswith('s') and not w.endswith('ss') and not w.endswith('us'):
                w = w[:-1]
            normalized.append(w)
        return ' '.join(normalized)

    async def merge_similar_clusters(
        self,
        cluster_report: pd.DataFrame,
        df_processed: pd.DataFrame,
        logs: List[str] = None
    ) -> tuple:
        """
        Merges clusters with the same normalized name (case + plural insensitive)
        within the same ticket_type. Uses groupby().agg() for clean aggregation.
        
        Returns:
            (merged_cluster_report, updated_df_processed)
        """
        if cluster_report.empty or len(cluster_report) < 2:
            return cluster_report, df_processed

        if logs is not None:
            logs.append("  - Checking for duplicate cluster names...")
        logger.info(f"Merge step: checking {len(cluster_report)} clusters for name-based duplicates...")

        # 1. Add normalized merge key (normalized_name + ticket_type)
        cluster_report = cluster_report.copy()
        cluster_report['_merge_key'] = (
            cluster_report['usecase_name'].apply(self._normalize_cluster_name)
            + '||' + cluster_report['ticket_type'].astype(str)
        )

        # 2. Check if any merging is needed
        duplicate_mask = cluster_report.duplicated(subset='_merge_key', keep=False)
        if not duplicate_mask.any():
            if logs is not None:
                logs.append("  - No duplicate cluster names found.")
            logger.info("Merge step: no name-based duplicates found.")
            cluster_report = cluster_report.drop(columns='_merge_key')
            return cluster_report, df_processed

        num_duplicates = int(duplicate_mask.sum())
        num_groups = cluster_report[duplicate_mask]['_merge_key'].nunique()
        logger.info(f"Merge step: found {num_duplicates} clusters collapsing into {num_groups} groups.")

        # 3. Sort by total_tickets descending so 'first' picks the largest cluster
        cluster_report = cluster_report.sort_values('total_tickets', ascending=False)

        # 4. Capture ID groups for remapping (before groupby collapses rows)
        id_groups = cluster_report.groupby('_merge_key')['usecase_id'].apply(list).to_dict()

        # 5. Build ID remap: non-primary IDs -> primary (first = largest)
        id_remap = {}
        for ids in id_groups.values():
            if len(ids) > 1:
                primary_id = ids[0]
                for old_id in ids[1:]:
                    id_remap[old_id] = primary_id

        # 6. Aggregate with groupby
        agg_dict = {
            'usecase_name': 'first',
            'technical_description': 'first',
            'ticket_type': 'first',
            'usecase_id': 'first',
        }
        for col in ['total_tickets', 'total_effort_hours', 'tickets_per_month', 'efforts_required_per_month_in_hours']:
            if col in cluster_report.columns:
                agg_dict[col] = 'sum'

        merged = cluster_report.groupby('_merge_key', sort=False).agg(agg_dict).reset_index(drop=True)

        # 7. Remap usecase_ids in individual tickets DataFrame
        if id_remap:
            remap_mask = df_processed['usecase_id'].isin(id_remap.keys())
            if remap_mask.any():
                df_processed.loc[remap_mask, 'usecase_id'] = (
                    df_processed.loc[remap_mask, 'usecase_id'].map(id_remap)
                )
                logger.info(f"Remapped {int(remap_mask.sum())} individual tickets to merged cluster IDs.")

        reduced_by = len(cluster_report) - len(merged)
        if logs is not None:
            logs.append(f"✅ Merged {num_duplicates} duplicate clusters into {num_groups} groups (reduced by {reduced_by}). Final: {len(merged)} clusters.")
        logger.info(f"Merge step complete. {len(cluster_report)} -> {len(merged)} clusters (reduced by {reduced_by}).")

        return merged, df_processed

    async def generate_final_summary(
        self, 
        cluster_report: pd.DataFrame, 
        logs: List[str] = None
    ) -> str:
        
        if logs is not None: logs.append("Generating the final executive summary...")
        logger.info("Generating final executive summary...")
        
        if cluster_report.empty:
            return "No usecases were identified in the provided data."
        report_string = cluster_report.to_string(index=False)
        
        SYSTEM_PROMPT_SUMMARY = """
You are a Senior Strategic IT Analyst. Your goal is to write a high-impact Executive Summary for the CIO based on the provided Ticket Use Case Report.

**OBJECTIVE:** Analyze the usecases to identify the biggest operational risks, resource drains, and systemic failures.

**OUTPUT FORMAT:**
Write a structured report in Markdown using the following sections:

### Executive Overview
[Write 2-3 sentences summarizing the overall volume and primary themes of the month. Mention if there is a significant amount of "Noise" or vague tickets.]

### Top Resource Drains (High Effort)
* **[Use Case Name]:** [Mention monthly ticket count] tickets/month requiring approx [Monthly Effort Hours] hours/month.
    * *Impact:* [Briefly explain why this is a burden, e.g., "High volume of manual resets," "Complex server alerts."]
* [Select the next highest effort cluster...]

### Recurring Technical Failures
* Identify specific technologies (e.g., Azure, Cisco, SAP) that appear across multiple clusters or have high counts.
* Highlight if specific components are failing (e.g., "Outlook Profiles," "VPN Certificates").

### Strategic Recommendations
* Suggest 2-3 high-value actions. 
    * *Example:* "Initiate a root cause analysis for [Cluster Name] due to high recurrence."
    * *Example:* "Consider automation for [Cluster Name] to reduce manual effort."
    * *Example:* "Improve end-user training for [Topic] to reduce 'How-to' requests."

**DATA:**
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_SUMMARY),
            ("user", f"Here is the Cluster Data:\n\n{report_string}")
        ])
        chain = prompt | self.llm
        try:
            result = await chain.ainvoke({})
            return result.content
        except Exception as e:
            logger.error(f"Final summary failed: {e}")
            return f"Error generating summary: {e}"
        
    async def identify_noise_clusters(
        self, 
        cluster_report: pd.DataFrame, 
        logs: List[str] = None
    ) -> List[int]:
        """
        Analyzes the *names* of the generated clusters. 
        If a name implies vague data (e.g., "Miscellaneous", "General Errors"), 
        returns the usecase_ids that should be marked as noise (-1).
        """
        if cluster_report.empty:
            return []

        if logs is not None: logs.append("  - Auditing cluster names for noise...")
        logger.info("Auditing cluster names for noise...")

        candidates = cluster_report[['usecase_id', 'usecase_name']].to_dict(orient='records')
        candidates_str = "\n".join([f"ID {x['usecase_id']}: {x['usecase_name']}" for x in candidates])

        SYSTEM_PROMPT_AUDIT = """
You are a Quality Assurance Auditor for IT Ticket Analysis. 
You are given a list of "Cluster Names" generated from a grouping algorithm.

**YOUR TASK:**
Identify which of these names represent **"Noise"**, **"Vague Data"**, or **"Insufficient Information"**.
A "Noise" cluster is one where the name indicates we don't know the actual technical issue.

**CRITERIA FOR NOISE:**
- Names containing "General", "Miscellaneous", "Other", "Unknown", "Unspecified".
- Names that describe the *process* not the *tech* (e.g., "Tickets Closed", "User Responded").
- Names indicating lack of data (e.g., "Insufficient Description", "Vague Requests").

**CRITERIA FOR VALID (KEEP):**
- Specific technologies (e.g., "Outlook", "Azure", "VPN").
- Specific errors (e.g., "Login Failed", "CPU High").

**OUTPUT:**
Return a JSON object with a single key "noise_ids" containing the list of IDs to reject.
Example: {{"noise_ids": [12, 45]}}
If all are valid, return {{"noise_ids": []}}.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_AUDIT),
            ("user", f"Here are the clusters:\n{candidates_str}")
        ])
        
        chain = prompt | self.llm | JsonOutputParser(pydantic_object=NoiseAudit)

        try:
            result = await chain.ainvoke({})
            noise_ids = result.get("noise_ids", []) if result else []
            logger.info(f"Auditor identified {len(noise_ids)} noise clusters: {noise_ids}")
            return noise_ids
        except Exception as e:
            logger.error(f"Cluster audit failed: {e}")
            return []

    async def run_full_analysis(
        self, 
        df: pd.DataFrame, 
        schema: SheetSchema, 
        logs: List[str] = None,
        duration_months: int = 1,
        ticket_type_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Runs the full analysis pipeline *after* schema is confirmed.
        """
        if logs is not None: logs.append("Full analysis pipeline initiated...")
        logger.info("Full analysis pipeline initiated...")
        
        df_processed = df.copy()
        
        id_col = schema.id_column
        summary_col = schema.summary_column
        res_col = schema.resolution_column

        if id_col not in df_processed.columns:
            raise ValueError(f"ID column '{id_col}' not found in DataFrame.")
        if summary_col not in df_processed.columns:
            raise ValueError(f"Summary column '{summary_col}' not found in DataFrame.")

        df_processed['ticket_type_prefix'] = df_processed[id_col].astype(str).str.extract(r'^([A-Za-z]+)')[0].str.upper()
        
        # Build run-specific type and effort maps
        run_type_map = dict(self.TICKET_TYPE_MAP)
        run_effort_map = dict(self.EFFORT_MAP)
        
        if ticket_type_config:
            for prefix, config in ticket_type_config.items():
                if prefix == "NUMERIC":
                    continue  # Handle numeric IDs separately below
                type_name = config.get("type_name", prefix)
                effort = config.get("effort_hours", 0)
                run_type_map[prefix] = type_name
                run_effort_map[type_name] = effort
                logger.info(f"Dynamic ticket type registered: {prefix} -> {type_name} ({effort} hrs)")
        
        # Map known + dynamically registered prefixes
        df_processed['ticket_type'] = df_processed['ticket_type_prefix'].map(run_type_map)
        
        # Handle NUMERIC tickets (no alphabetical prefix)
        if ticket_type_config and "NUMERIC" in ticket_type_config:
            numeric_config = ticket_type_config["NUMERIC"]
            numeric_mask = df_processed['ticket_type_prefix'].isna()
            if numeric_mask.any():
                type_name = numeric_config.get("type_name", "General Ticket")
                effort = numeric_config.get("effort_hours", 0)
                df_processed.loc[numeric_mask, 'ticket_type'] = type_name
                run_effort_map[type_name] = effort
                logger.info(f"Numeric ticket type registered: NUMERIC -> {type_name} ({effort} hrs)")
        
        # Drop rows where ticket type is still unknown
        df_processed = df_processed[df_processed['ticket_type'].notna()].copy()
        df_processed['estimated_hours'] = df_processed['ticket_type'].map(run_effort_map).fillna(0)
        
        if df_processed.empty:
            raise ValueError("No valid ticket types found. Ensure the ID column contains recognizable ticket prefixes.")
        
        logger.info("Preparing text for summarization...")
        
        # Fill NA in summary
        df_processed[summary_col] = df_processed[summary_col].fillna('')
        
        # Combine with resolution if it exists
        if res_col and res_col in df_processed.columns:
            df_processed[res_col] = df_processed[res_col].fillna('')
            combined_text_list = (
                df_processed[summary_col].astype(str) + " | " + df_processed[res_col].astype(str)
            ).tolist()
        else:
            combined_text_list = df_processed[summary_col].astype(str).tolist()

        # Pass the COMBINED list to the summarizer
        summaries = await self.generate_rich_summaries(
            combined_text_list, 
            logs = logs
        )
        df_processed['rich_summary'] = summaries
        

        embed_msg = f"Creating embeddings for {len(df_processed)} tickets..."
        if logs is not None: logs.append(embed_msg)
        logger.info(embed_msg)
        
        embeddings = await self.embedder.aembed_documents(df_processed['rich_summary'].tolist())
        
        if logs is not None: logs.append("✅ Embeddings complete.")
        logger.info("Embeddings complete.")

        df_clustered = await asyncio.to_thread(
            self.cluster_data,
            df_processed, 
            np.array(embeddings), 
            logs
        )
        
        #Initialize cluster id with -1 for all rows
        if 'usecase_id' not in df_processed.columns:
             df_processed['usecase_id'] = -1

        # Merge the cluster IDs back into the main DF
        df_processed.update(df_clustered[['usecase_id']])
        
        df_cluster_report = await self.generate_cluster_names(
            df_processed, 
            logs = logs,
            duration_months=duration_months,
            effort_map=run_effort_map
        )
        
        # --- MERGE DUPLICATE CLUSTERS ---
        if not df_cluster_report.empty:
            df_cluster_report, df_processed = await self.merge_similar_clusters(
                df_cluster_report, df_processed, logs=logs
            )

        # --- AUDIT THE NAMES ---
        if not df_cluster_report.empty:
            noise_ids = await self.identify_noise_clusters(
                df_cluster_report, 
                logs = logs
            )
            
            if noise_ids:
                if logs is not None: logs.append(f"⚠️ Re-classified {len(noise_ids)} clusters as 'Noise' based on vague descriptions.")
                
                # 1. Update the Report: Remove these rows
                df_cluster_report = df_cluster_report[~df_cluster_report['usecase_id'].isin(noise_ids)].copy()
                
                # 2. Update the Main DF: Set their ID to -1 (Noise)
                df_processed.loc[df_processed['usecase_id'].isin(noise_ids), 'usecase_id'] = -1

        if not df_cluster_report.empty:
            report_cols = [
                "usecase_name","technical_description", "ticket_type", 
                "tickets_per_month", "efforts_required_per_month_in_hours", "usecase_id"
            ]
            final_report_cols = [col for col in report_cols if col in df_cluster_report.columns]
            df_cluster_report = df_cluster_report[final_report_cols]

        final_summary_text = await self.generate_final_summary(
            df_cluster_report, 
            logs = logs
        )
        
        return {
            "individual_tickets": df_processed,
            "cluster_report": df_cluster_report,
            "executive_summary": final_summary_text
        }