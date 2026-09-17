# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 9: S2P Process Twin AI Copilot Agent — Creation & Deployment
# MAGIC 
# MAGIC **Purpose**: Builds the S2P Supply Chain Copilot AI agent powered by Databricks Foundation Models (Llama 3.3 70B Instruct), equips it with the 5 native Unity Catalog tools registered in Notebook 08, logs it to MLflow, and deploys it as a Model Serving endpoint named `s2p-procurement-copilot`.

# COMMAND ----------

# MAGIC %pip install --upgrade "langgraph>=1.1.5" "langgraph-prebuilt>=1.0.9" databricks-langchain langchain langchain-core mlflow databricks-sdk unitycatalog-langchain[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentCreation")

CATALOG = "s2p_twin"
SCHEMA = "delayed_shipment"
AGENT_MODEL_NAME = f"{CATALOG}.{SCHEMA}.s2p_procurement_copilot"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Define the Agent System Prompt
# MAGIC 
# MAGIC Shapes the agent's procurement domain expertise, tool routing rules, and response structure.

# COMMAND ----------

SYSTEM_PROMPT = """You are the **S2P Supply Chain Copilot** — an expert procurement AI analyst at Motiveminds Consulting, powered by a real-time Process Digital Twin running on SAP Business Data Cloud (Databricks).

## Your Tools (Always invoke the right tool based on the user's intent):
1. **get_po_analysis(po_number_input)**:
   - **CALL THIS TOOL** whenever the user mentions, asks about, looks up, or asks to analyze a Purchase Order (e.g. `PO-4500103`, `PO-4500045`, etc.).
   - Returns complete PO details: order date, delivery date, total value in USD, vendor name, vendor OTD%, material description, material criticality, shipment ID, tracking status, delay days, delay reason, carrier name, ML recommendation priority, predicted delay days, and SHAP risk drivers.

2. **get_shipment_analysis(shipment_id_input)**:
   - **CALL THIS TOOL** whenever the user asks about a specific shipment (e.g. `SHP-90045`, `SHP-90078`, etc.).
   - Returns tracking status, delay days, delay reason, carrier reliability score, shipping route risk factor, linked PO, and prescriptive mitigations.

3. **get_morning_risk_scan()**:
   - **CALL THIS TOOL** whenever the user asks for a morning risk scan, risk briefing, critical shipment alerts, total value at risk, or an executive overview of active supply chain anomalies.
   - Returns counts of at-risk shipments, total dollar exposure, critical/high breakdowns, and the top delayed shipments.

4. **compare_vendor_performance(vendor_name_filter)**:
   - **CALL THIS TOOL** whenever the user asks to compare vendors, inspect supplier performance, evaluate on-time delivery rates (OTD%), check vendor risk scores, or explore dual-sourcing options.
   - Pass `'ALL'` to get all vendors, or pass a specific vendor name (e.g. `'Meridian'`, `'Apex'`).

5. **simulate_whatif_scenario(scenario, pct_rerouted, cost_multiplier, split_pct, secondary_otd, buffer_days, holding_cost_pct, consolidation_pct)**:
   - **CALL THIS TOOL** whenever the user asks for What-If simulations or policy optimization:
     - `carrier_switch`: reroute delayed sea freight to air freight (params: `pct_rerouted`, `cost_multiplier`)
     - `dual_source`: split order volume to secondary supplier (params: `split_pct`, `secondary_otd`)
     - `buffer_stock`: add safety buffer days (params: `buffer_days`, `holding_cost_pct`)
     - `vendor_consolidation`: consolidate volume to top suppliers (params: `consolidation_pct`)
   - Returns projected OTD%, average delay days, projected costs, and P5/Mean/P95 confidence intervals.

## Response Formatting Guidelines:
- **ALWAYS call the appropriate tool first** to fetch real data. Never fabricate PO, shipment, or vendor data.
- When answering PO queries, structure your response clearly:
  - **PO Header & Overview**: PO number, vendor, material description, criticality, total value ($).
  - **Shipment & Delivery Status**: Tracking status, carrier, expected vs actual delivery, delay days.
  - **Risk Assessment & Explainability**: Delay priority (CRITICAL/HIGH/MEDIUM/LOW), SHAP risk drivers explaining WHY it is delayed.
  - **Recommended Mitigation**: Specific action steps (e.g., partial air freight expedite, alternative sourcing) with rationale.
- Use currency formatting (`$`) and percentages (`%`).
- Be concise, professional, and directly answer the question asked.
"""

logger.info(f"System prompt defined ({len(SYSTEM_PROMPT)} chars)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Assemble the Agent with UC Function Tools

# COMMAND ----------

from databricks_langchain import ChatDatabricks, UCFunctionToolkit
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LLM Endpoint — change if using a different foundation model
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT,
    temperature=0.1
)

logger.info(f"LLM initialized: {LLM_ENDPOINT}")

# Load the 5 native Unity Catalog tools registered in Notebook 08
toolkit = UCFunctionToolkit(function_names=[
    f"{CATALOG}.{SCHEMA}.get_po_analysis",
    f"{CATALOG}.{SCHEMA}.get_shipment_analysis",
    f"{CATALOG}.{SCHEMA}.get_morning_risk_scan",
    f"{CATALOG}.{SCHEMA}.compare_vendor_performance",
    f"{CATALOG}.{SCHEMA}.simulate_whatif_scenario"
])
tools = toolkit.tools

logger.info(f"Loaded {len(tools)} UC function tools:")
for t in tools:
    logger.info(f"  • {t.name}")

# Agent Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Assemble Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True
)

logger.info("✅ Agent assembled successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Test the Agent Interactively
# MAGIC 
# MAGIC Testing the exact queries: PO analysis, morning risk scan, vendor comparison, and simulation.

# COMMAND ----------

# Test 1: PO Analysis (The exact query reported by the user)
test_response_po = agent_executor.invoke({
    "input": "Analyze for PO - PO-4500103"
})
print("─" * 60)
print("PO ANALYSIS RESPONSE:")
print(test_response_po["output"])

# COMMAND ----------

# Test 2: Morning Risk Scan
test_response_risk = agent_executor.invoke({
    "input": "Perform a morning risk scan and show all critical shipments at risk."
})
print("─" * 60)
print("RISK SCAN RESPONSE:")
print(test_response_risk["output"])

# COMMAND ----------

# Test 3: Vendor Comparison
test_response_vendor = agent_executor.invoke({
    "input": "Compare Meridian Components vs Apex Industrial on OTD and risk score."
})
print("─" * 60)
print("VENDOR COMPARISON RESPONSE:")
print(test_response_vendor["output"])

# COMMAND ----------

# Test 4: What-If Simulation
test_response_sim = agent_executor.invoke({
    "input": "Run a carrier_switch simulation with 40% rerouted to air freight at 3x cost multiplier."
})
print("─" * 60)
print("SIMULATION RESPONSE:")
print(test_response_sim["output"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Register Agent in Unity Catalog & Log to MLflow

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksFunction
)

# Declare UC Function dependencies for automatic governance
resources = [
    DatabricksFunction(function_name=f"{CATALOG}.{SCHEMA}.get_po_analysis"),
    DatabricksFunction(function_name=f"{CATALOG}.{SCHEMA}.get_shipment_analysis"),
    DatabricksFunction(function_name=f"{CATALOG}.{SCHEMA}.get_morning_risk_scan"),
    DatabricksFunction(function_name=f"{CATALOG}.{SCHEMA}.compare_vendor_performance"),
    DatabricksFunction(function_name=f"{CATALOG}.{SCHEMA}.simulate_whatif_scenario"),
]

experiment_name = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/s2p_copilot_experiment"
mlflow.set_experiment(experiment_name)
logger.info(f"MLflow experiment: {experiment_name}")

with mlflow.start_run(run_name="s2p_copilot_v2_native"):
    model_info = mlflow.langchain.log_model(
        lc_model=agent_executor,
        artifact_path="s2p_copilot_agent",
        registered_model_name=AGENT_MODEL_NAME,
        resources=resources,
        input_example={"input": "Analyze for PO - PO-4500103"},
        extra_pip_requirements=[
            "databricks-langchain>=0.1.0",
            "unitycatalog-langchain[databricks]>=0.1.0",
            "databricks-sdk>=0.35.0",
            "langchain>=0.3.0",
            "langchain-core>=0.3.0",
            "langgraph>=1.1.5",
            "langgraph-prebuilt>=1.0.9"
        ]
    )
    logger.info(f"Model registered: {AGENT_MODEL_NAME}")
    logger.info(f"Model URI: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Deploy / Update the Model Serving Endpoint

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput
)

w = WorkspaceClient()

# Get the latest model version
latest_version = max(
    w.model_registry.search_model_versions(f"name='{AGENT_MODEL_NAME}'"),
    key=lambda v: int(v.version)
).version

logger.info(f"Deploying model version {latest_version} of {AGENT_MODEL_NAME}")

ENDPOINT_NAME = "s2p-procurement-copilot"

try:
    existing = w.serving_endpoints.get(ENDPOINT_NAME)
    logger.info(f"Endpoint '{ENDPOINT_NAME}' exists. Updating to v{latest_version}...")
    
    w.serving_endpoints.update_config(
        name=ENDPOINT_NAME,
        served_entities=[
            ServedEntityInput(
                entity_name=AGENT_MODEL_NAME,
                entity_version=latest_version,
                workload_size="Small",
                scale_to_zero_enabled=True
            )
        ]
    )
except Exception:
    logger.info(f"Creating new endpoint: {ENDPOINT_NAME}")
    
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=AGENT_MODEL_NAME,
                    entity_version=latest_version,
                    workload_size="Small",
                    scale_to_zero_enabled=True
                )
            ]
        )
    )

print(f"""
{'='*60}
  S2P PROCUREMENT COPILOT — SERVING ENDPOINT UPDATED
{'='*60}
  Endpoint:         {ENDPOINT_NAME}
  Model:            {AGENT_MODEL_NAME} v{latest_version}
  Tools:            5 Native UC Tools
  Scale to Zero:    Enabled
  
  Check status in: Databricks -> Serving -> {ENDPOINT_NAME}
{'='*60}
""")
