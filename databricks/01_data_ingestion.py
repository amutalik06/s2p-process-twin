# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1: Data Ingestion (Unity Catalog)
# MAGIC 
# MAGIC **Purpose**: Ingests the 10 S2P Process Twin CSV flat files from Unity Catalog Volumes into Delta tables.
# MAGIC **Catalog**: `s2p_twin`
# MAGIC **Schema**: `delayed_shipment`
# MAGIC **Volume**: `/Volumes/s2p_twin/delayed_shipment/raw_data/`

# COMMAND ----------

import os
import pyspark.sql.functions as F
from pyspark.sql.types import *
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataIngestion")

CATALOG = "s2p_twin"
SCHEMA = "delayed_shipment"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_data/"

# Ensure Catalog and Schema exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

logger.info(f"Target Catalog: {CATALOG}, Schema: {SCHEMA}, Volume: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## List of CSV Files to Ingest

# COMMAND ----------

# Mapping of Volume CSV files to Target Delta Table names
files_to_tables = {
    "DIM_VENDOR.csv": "dim_vendor",
    "DIM_MATERIAL.csv": "dim_material",
    "DIM_PLANT.csv": "dim_plant",
    "DIM_CARRIER.csv": "dim_carrier",
    "DIM_SHIPPING_ROUTE.csv": "dim_shipping_route",
    "FCT_PURCHASE_ORDER.csv": "fct_purchase_order",
    "FCT_SHIPMENT.csv": "fct_shipment",
    "FCT_SHIPMENT_EVENT.csv": "fct_shipment_event",
    "FCT_GOODS_RECEIPT.csv": "fct_goods_receipt",
    "FCT_VENDOR_PERFORMANCE.csv": "fct_vendor_performance"
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestion Function

# COMMAND ----------

def ingest_file(filename, table_name):
    logger.info(f"Processing: {filename} -> Table: {table_name}")
    file_path = f"{VOLUME_PATH}{filename}"
    
    try:
        # Read CSV with header and automatic schema inference
        df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .option("mode", "PERMISSIVE") \
            .load(file_path)
            
        row_count = df.count()
        if row_count == 0:
            logger.warning(f"File {filename} is empty. Skipping table write.")
            return False
            
        # Standardize column names to lowercase for clean SQL querying
        for col_name in df.columns:
            clean_col = col_name.strip().lower()
            df = df.withColumnRenamed(col_name, clean_col)
            
        # Add metadata audit columns
        df = df.withColumn("_load_timestamp", F.current_timestamp()) \
               .withColumn("_source_file", F.lit(filename))
               
        # Save as Delta Table in Unity Catalog
        full_table_name = f"{CATALOG}.{SCHEMA}.{table_name}"
        df.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(full_table_name)
            
        logger.info(f" Successfully ingested {filename} into {full_table_name} ({row_count} rows)")
        return True
        
    except Exception as e:
        logger.error(f" Failed to ingest {filename}: {str(e)}")
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Ingestion for All Tables

# COMMAND ----------

results = {}
for file_name, table_name in files_to_tables.items():
    success = ingest_file(file_name, table_name)
    results[table_name] = "SUCCESS" if success else "FAILED"

print("\n" + "="*50)
print("           INGESTION SUMMARY")
print("="*50)
for tbl, status in results.items():
    print(f"Table: {tbl:<28} | Status: {status}")
print("="*50)

# Verify table count in schema
table_count = spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").count()
print(f"\nTotal Delta Tables in {CATALOG}.{SCHEMA}: {table_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Data Preview

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.fct_shipment").limit(5))
