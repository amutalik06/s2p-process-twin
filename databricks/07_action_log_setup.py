# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 7: Action Log Table & Initial Setup
# MAGIC 
# MAGIC **Purpose**: Creates the `action_log` Delta table that the Live Process Twin UI writes confirmed actions to.
# MAGIC This table closes the loop between AI recommendations and human decisions.

# COMMAND ----------

CATALOG = "s2p_twin"
SCHEMA = "delayed_shipment"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Action Log Table

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.action_log (
    action_id STRING NOT NULL COMMENT 'Unique action execution ID',
    recommendation_id STRING COMMENT 'Links back to the recommendation that triggered this action',
    shipment_id STRING COMMENT 'Affected shipment ID',
    po_number STRING COMMENT 'Affected PO number',
    action_type STRING COMMENT 'Category: EXPEDITE, SPLIT_SOURCE, ESCALATE, CUSTOMS, MONITOR, ACKNOWLEDGE',
    action_label STRING COMMENT 'Human-readable action description',
    executed_by STRING COMMENT 'User who confirmed the action',
    executed_at TIMESTAMP COMMENT 'When the action was confirmed',
    notes STRING COMMENT 'Optional free-text notes from the user',
    cost_impact DOUBLE COMMENT 'Estimated cost impact in USD',
    priority STRING COMMENT 'Priority at time of action: CRITICAL, HIGH, MEDIUM, LOW',
    status STRING DEFAULT 'EXECUTED' COMMENT 'EXECUTED, PENDING, REVERTED',
    outcome STRING COMMENT 'Outcome after follow-up: RESOLVED, PARTIALLY_RESOLVED, UNRESOLVED, PENDING'
)
USING DELTA
COMMENT 'Closed-loop action log for the S2P Process Twin. Records every human decision taken on AI recommendations.'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")

print("✅ action_log table created successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Insert Sample Historical Actions (for Audit Trail Demo)

# COMMAND ----------

spark.sql(f"""
INSERT INTO {CATALOG}.{SCHEMA}.action_log VALUES
    ('ACT-2026-0001', 'REC-2026-00101', 'SHP-90045', 'PO-4500045', 'EXPEDITE', 'Expedited via Air Freight (DHL Express)', 'Jane Smith', '2026-08-20T09:15:00Z', 'Approved emergency air shipment for hydraulic pumps', 85000.0, 'CRITICAL', 'EXECUTED', 'RESOLVED'),
    ('ACT-2026-0002', 'REC-2026-00102', 'SHP-90078', 'PO-4500078', 'SPLIT_SOURCE', 'Split order 60/40 with Apex Industrial', 'Hans Weber', '2026-08-21T14:30:00Z', 'Secondary vendor confirmed capacity for 40% volume', 12000.0, 'CRITICAL', 'EXECUTED', 'PARTIALLY_RESOLVED'),
    ('ACT-2026-0003', 'REC-2026-00103', 'SHP-90034', 'PO-4500034', 'CUSTOMS', 'Engaged fast-track customs broker at Chennai', 'Priya Nair', '2026-08-22T11:00:00Z', 'Broker estimates 2-day clearance', 4500.0, 'HIGH', 'EXECUTED', 'RESOLVED'),
    ('ACT-2026-0004', 'REC-2026-00104', 'SHP-90091', 'PO-4500091', 'ESCALATE', 'Escalated to VP Supply Chain', 'Mike Chen', '2026-08-23T16:45:00Z', 'VP authorized vendor penalty clause activation', 0.0, 'HIGH', 'EXECUTED', 'PENDING')
""")

print("✅ 4 sample action records inserted.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Tables

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.action_log ORDER BY executed_at DESC"))
