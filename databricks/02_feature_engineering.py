# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 2: Feature Engineering (ML Feature Store)
# MAGIC 
# MAGIC **Purpose**: Joins Fact and Dimension Delta tables to build ML-ready features for delay prediction.
# MAGIC **Catalog**: `s2p_twin`
# MAGIC **Schema**: `delayed_shipment`
# MAGIC **Target Table**: `s2p_twin.delayed_shipment.shipment_features`

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureEngineering")

CATALOG = "s2p_twin"
SCHEMA = "delayed_shipment"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Ingested Delta Tables

# COMMAND ----------

try:
    fct_shipment = spark.table(f"{CATALOG}.{SCHEMA}.fct_shipment")
    fct_po = spark.table(f"{CATALOG}.{SCHEMA}.fct_purchase_order")
    dim_vendor = spark.table(f"{CATALOG}.{SCHEMA}.dim_vendor")
    dim_material = spark.table(f"{CATALOG}.{SCHEMA}.dim_material")
    dim_route = spark.table(f"{CATALOG}.{SCHEMA}.dim_shipping_route")
    dim_carrier = spark.table(f"{CATALOG}.{SCHEMA}.dim_carrier")
    dim_plant = spark.table(f"{CATALOG}.{SCHEMA}.dim_plant")
    
    logger.info("Successfully loaded all base Delta tables.")
except Exception as e:
    logger.error(f"Error loading base tables: {str(e)}")
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join Tables into Base Analytical Matrix

# COMMAND ----------

# Join Shipments with PO and Dimensions
base_df = fct_shipment.alias("s") \
    .join(fct_po.alias("po"), F.col("s.po_number") == F.col("po.po_number"), "left") \
    .join(dim_vendor.alias("v"), F.col("po.vendor_id") == F.col("v.vendor_id"), "left") \
    .join(dim_material.alias("m"), F.col("po.material_id") == F.col("m.material_id"), "left") \
    .join(dim_route.alias("r"), F.col("s.route_id") == F.col("r.route_id"), "left") \
    .join(dim_carrier.alias("c"), F.col("s.carrier_id") == F.col("c.carrier_id"), "left") \
    .join(dim_plant.alias("p"), F.col("po.plant_id") == F.col("p.plant_id"), "left") \
    .select(
        F.col("s.shipment_id"),
        F.col("s.po_number"),
        F.col("po.vendor_id"),
        F.col("v.vendor_name"),
        F.col("po.material_id"),
        F.col("m.material_desc"),
        F.col("m.criticality"),
        F.col("m.safety_stock_days"),
        F.col("po.plant_id"),
        F.col("p.plant_name"),
        F.col("s.carrier_id"),
        F.col("c.carrier_name"),
        F.col("s.route_id"),
        F.col("s.transport_mode"),
        F.col("s.ship_date"),
        F.col("s.original_eta"),
        F.col("s.revised_eta"),
        F.col("s.actual_delivery_date"),
        F.col("s.delay_days").cast("double"),
        F.col("s.delay_reason"),
        F.col("s.shipment_status"),
        F.col("po.order_date"),
        F.col("po.quantity"),
        F.col("po.unit_price"),
        F.col("po.total_value"),
        F.col("po.currency"),
        F.col("v.on_time_delivery_pct"),
        F.col("v.risk_score").alias("vendor_risk_score"),
        F.col("v.avg_lead_time_days").alias("vendor_avg_lead_time"),
        F.col("r.risk_factor").alias("route_risk_factor"),
        F.col("c.reliability_score").alias("carrier_reliability_score")
    )

logger.info(f"Base joined dataset count: {base_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Transformations & Feature Engineering

# COMMAND ----------

# Target Binary Variable
base_df = base_df.withColumn(
    "is_delayed", 
    F.when(F.coalesce(F.col("delay_days"), F.lit(0)) > 0, 1).otherwise(0)
)

# 1. vendor_historical_otd (Normalized 0-1)
base_df = base_df.withColumn(
    "vendor_historical_otd", 
    F.coalesce(F.col("on_time_delivery_pct") / 100.0, F.lit(0.85))
)

# 2. route_avg_delay
w_route = Window.partitionBy("route_id")
base_df = base_df.withColumn(
    "route_avg_delay", 
    F.coalesce(F.avg("delay_days").over(w_route), F.lit(2.5))
)

# 3. material_criticality_score (Numeric: HIGH=3, MEDIUM=2, LOW=1)
base_df = base_df.withColumn(
    "material_criticality_score",
    F.when(F.upper(F.col("criticality")) == "HIGH", 3.0)
     .when(F.upper(F.col("criticality")) == "MEDIUM", 2.0)
     .otherwise(1.0)
)

# 4. seasonal_risk_factor (Month-based)
base_df = base_df.withColumn("ship_month", F.month(F.coalesce(F.col("ship_date"), F.col("order_date"))))
base_df = base_df.withColumn(
    "seasonal_risk_factor",
    F.when(F.col("ship_month").isin([11, 12, 1]), 1.5)
     .when(F.col("ship_month").isin([7, 8]), 1.2)
     .otherwise(1.0)
)

# 5. vendor_lead_time_variance
w_vendor = Window.partitionBy("vendor_id")
base_df = base_df.withColumn(
    "vendor_lead_time_variance", 
    F.coalesce(F.stddev("delay_days").over(w_vendor), F.lit(1.2))
)

# 6. days_since_order
base_df = base_df.withColumn(
    "days_since_order", 
    F.coalesce(F.datediff(F.current_date(), F.col("order_date")).cast("double"), F.lit(10.0))
)

# 7. expected_transit_duration
base_df = base_df.withColumn(
    "expected_transit_duration", 
    F.coalesce(F.datediff(F.col("original_eta"), F.col("ship_date")).cast("double"), F.lit(14.0))
)

# 8. vendor_spend_concentration
w_vendor_spend = Window.partitionBy("vendor_id")
w_all = Window.partitionBy()
base_df = base_df.withColumn("vendor_total_spend", F.sum("total_value").over(w_vendor_spend)) \
                 .withColumn("overall_total_spend", F.sum("total_value").over(w_all)) \
                 .withColumn("vendor_spend_concentration", F.coalesce(F.col("vendor_total_spend") / F.col("overall_total_spend"), F.lit(0.05)))

# 9. carrier_reliability_for_route
base_df = base_df.withColumn(
    "carrier_reliability_for_route", 
    F.coalesce(F.col("carrier_reliability_score"), F.lit(0.88))
)

# 10. buffer_stock
base_df = base_df.withColumn(
    "buffer_stock", 
    F.coalesce(F.col("safety_stock_days").cast("double"), F.lit(10.0))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Unity Catalog Feature Store Table

# COMMAND ----------

# Clean nulls for all feature columns
feature_columns = [
    "vendor_historical_otd", "route_avg_delay", "material_criticality_score",
    "seasonal_risk_factor", "vendor_lead_time_variance", "days_since_order",
    "expected_transit_duration", "vendor_spend_concentration", "carrier_reliability_for_route",
    "delay_days", "is_delayed", "buffer_stock"
]

for col_name in feature_columns:
    base_df = base_df.fillna(0.0, subset=[col_name])

target_table = f"{CATALOG}.{SCHEMA}.shipment_features"
base_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

logger.info(f"Successfully saved {base_df.count()} feature rows to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Matrix Preview

# COMMAND ----------

display(spark.table(target_table).select(
    "shipment_id", "po_number", "vendor_name", "material_desc", 
    "is_delayed", "delay_days", "vendor_historical_otd", "material_criticality_score", "seasonal_risk_factor"
).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## On-Demand Feature Extraction for Dynamic Inference

# COMMAND ----------

def get_shipment_features_by_id(shipment_id):
    """
    Extracts the feature vector for a specific shipment ID from the feature store.
    Returns a dictionary of feature values ready for model inference.
    """
    row = spark.table(f"{CATALOG}.{SCHEMA}.shipment_features") \
        .filter(F.col("shipment_id") == shipment_id) \
        .toPandas()
    if len(row) == 0:
        return None
    return row.iloc[0].to_dict()

