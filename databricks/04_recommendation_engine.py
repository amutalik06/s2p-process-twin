# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 4: Recommendation Engine (Action Layer)
# MAGIC 
# MAGIC **Purpose**: Evaluates ML predictions with business priority rules and generates draft action recommendations.
# MAGIC **Catalog**: `s2p_twin`
# MAGIC **Schema**: `delayed_shipment`
# MAGIC **Input Table**: `s2p_twin.delayed_shipment.delay_predictions`
# MAGIC **Output Table**: `s2p_twin.delayed_shipment.recommendations`

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
import uuid
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RecommendationEngine")

CATALOG = "s2p_twin"
SCHEMA = "delayed_shipment"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Scored Predictions

# COMMAND ----------

preds_df = spark.table(f"{CATALOG}.{SCHEMA}.delay_predictions")
logger.info(f"Loaded {preds_df.count()} prediction records.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Rules Engine (Priority Scoring)

# COMMAND ----------

# Apply Multi-Tier Priority Logic
preds_with_priority = preds_df.withColumn(
    "priority",
    F.when(
        (F.col("predicted_delay_days") >= 8.0) & 
        (F.col("buffer_stock") <= 5.0) & 
        (F.col("material_criticality_score") >= 3.0),
        "CRITICAL"
    ).when(
        (F.col("predicted_delay_days") >= 5.0) | 
        ((F.col("predicted_delay_days") >= 3.0) & (F.col("buffer_stock") <= 3.0)),
        "HIGH"
    ).when(
        F.col("predicted_delay_days") >= 2.0,
        "MEDIUM"
    ).otherwise("LOW")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Action Generation & Formatting

# COMMAND ----------

def get_recommended_actions(priority):
    actions_map = {
        "CRITICAL": [
            {"action_id": "ACT-EXPEDITE-AIR", "label": "Expedite Partial via Air Freight", "is_default": True},
            {"action_id": "ACT-ALT-SOURCE", "label": "Emergency Source from Domestic Vendor", "is_default": False},
            {"action_id": "ACT-ESCALATE-VP", "label": "Escalate to Supply Chain VP", "is_default": False}
        ],
        "HIGH": [
            {"action_id": "ACT-SPLIT-ORDER", "label": "Split Source 40% with Alt Vendor", "is_default": True},
            {"action_id": "ACT-CUSTOMS-BROKER", "label": "Engage Fast-Track Customs Broker", "is_default": False},
            {"action_id": "ACT-ADJUST-SCHEDULE", "label": "Reschedule Production Assembly Line", "is_default": False}
        ],
        "MEDIUM": [
            {"action_id": "ACT-MONITOR-CARRIER", "label": "Contact Carrier for Real-Time GPS Tracking", "is_default": True},
            {"action_id": "ACT-CHECK-SAFETY-STOCK", "label": "Verify Safety Stock Buffer at Receiving Plant", "is_default": False}
        ],
        "LOW": [
            {"action_id": "ACT-MONITOR-ONLY", "label": "Standard Automated Tracking", "is_default": True},
            {"action_id": "ACT-ACKNOWLEDGE", "label": "Acknowledge & Close Alert", "is_default": False}
        ]
    }
    return json.dumps(actions_map.get(priority, actions_map["LOW"]))

actions_udf = F.udf(get_recommended_actions, StringType())

def generate_rec_id():
    return f"REC-2026-{uuid.uuid4().hex[:5].upper()}"

rec_id_udf = F.udf(generate_rec_id, StringType())

# Build Enriched Recommendations
recommendations_df = preds_with_priority \
    .withColumn("recommendation_id", rec_id_udf()) \
    .withColumn("status", F.lit("PENDING")) \
    .withColumn("confidence_score", F.round(F.col("predicted_is_delayed_proba"), 2)) \
    .withColumn("predicted_delay_days", F.round(F.col("predicted_delay_days"), 1)) \
    .withColumn(
        "summary", 
        F.concat(
            F.col("material_desc"), 
            F.lit(" from "), 
            F.col("vendor_name"), 
            F.lit(" (PO: "), 
            F.col("po_number"), 
            F.lit(") — predicted delay of "), 
            F.col("predicted_delay_days"), 
            F.lit(" days.")
        )
    ) \
    .withColumn("rationale", F.concat(F.lit("Top contributing risk drivers: "), F.col("top_shap_factors"))) \
    .withColumn("recommended_actions", actions_udf(F.col("priority"))) \
    .withColumn("assignee_role", F.lit("PROCUREMENT_OPS")) \
    .withColumn("created_at", F.current_timestamp())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Unity Catalog Recommendations Table

# COMMAND ----------

output_table = f"{CATALOG}.{SCHEMA}.recommendations"

final_cols = [
    "recommendation_id", "shipment_id", "po_number", "vendor_name", 
    "material_desc", "plant_name", "total_value", "currency",
    "priority", "status", "confidence_score", "predicted_delay_days", 
    "summary", "rationale", "recommended_actions", "assignee_role", "created_at"
]

available_cols = [c for c in final_cols if c in recommendations_df.columns]

recommendations_df.select(available_cols).write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(output_table)

total_recs = recommendations_df.count()
critical_recs = recommendations_df.filter("priority = 'CRITICAL'").count()
high_recs = recommendations_df.filter("priority = 'HIGH'").count()

print("="*50)
print(f"Generated {total_recs} recommendations into {output_table}")
print(f"  - CRITICAL Priority : {critical_recs}")
print(f"  - HIGH Priority     : {high_recs}")
print(f"  - MEDIUM/LOW        : {total_recs - (critical_recs + high_recs)}")
print("="*50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Actionable Recommendations Preview

# COMMAND ----------

display(spark.table(output_table).select(
    "recommendation_id", "priority", "confidence_score", "predicted_delay_days", "summary", "assignee_role"
).orderBy(F.when(F.col("priority") == "CRITICAL", 1).when(F.col("priority") == "HIGH", 2).otherwise(3)).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dynamic Recommendation Evaluation Function

# COMMAND ----------

def evaluate_recommendation_dynamic(predicted_delay_days, buffer_stock=10.0, criticality_score=2.0, critical_threshold=8.0, high_threshold=5.0):
    """
    Dynamically evaluates recommendation priority and actions when parameters/thresholds are changed in the UI.
    """
    if (predicted_delay_days >= critical_threshold) and (buffer_stock <= 5.0) and (criticality_score >= 3.0):
        priority = "CRITICAL"
    elif (predicted_delay_days >= high_threshold) or ((predicted_delay_days >= 3.0) and (buffer_stock <= 3.0)):
        priority = "HIGH"
    elif predicted_delay_days >= 2.0:
        priority = "MEDIUM"
    else:
        priority = "LOW"
        
    actions = json.loads(get_recommended_actions(priority))
    return {
        "priority": priority,
        "predicted_delay_days": predicted_delay_days,
        "buffer_stock": buffer_stock,
        "buffer_remaining": buffer_stock - predicted_delay_days,
        "recommended_actions": actions,
        "thresholds_used": {"critical": critical_threshold, "high": high_threshold}
    }

