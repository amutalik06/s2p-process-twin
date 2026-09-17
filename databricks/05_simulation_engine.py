# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 5: Simulation Engine (Horizon 4 What-If Simulator)
# MAGIC 
# MAGIC **Purpose**: Executes Monte Carlo simulations for supply chain policy adjustments (Carrier Switching, Dual Sourcing, Buffer Stock, Vendor Consolidation, Route Optimization).
# MAGIC **Catalog**: `s2p_twin`
# MAGIC **Schema**: `delayed_shipment`
# MAGIC **Output Table**: `s2p_twin.delayed_shipment.simulation_results`

# COMMAND ----------

import numpy as np
import json
import logging
from datetime import datetime
from pyspark.sql.types import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimulationEngine")

CATALOG = "s2p_twin"
SCHEMA = "delayed_shipment"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monte Carlo Helper Functions

# COMMAND ----------

def run_monte_carlo(base_value, std_dev, iterations=1000):
    samples = np.random.normal(base_value, max(0.01, std_dev), iterations)
    return {
        "mean": float(np.round(np.mean(samples), 2)),
        "p5": float(np.round(np.percentile(samples, 5), 2)),
        "p95": float(np.round(np.percentile(samples, 95), 2))
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulation Scenario Models

# COMMAND ----------

def carrier_switch_scenario(params):
    """What if we reroute delayed sea-freight shipments to air freight?"""
    pct = params.get('pct_rerouted', 50) / 100.0
    cost_mult = params.get('cost_multiplier', 3.0)
    base_delay = params.get('base_delay', 7.5)
    base_cost = params.get('base_cost', 1034700.0)
    base_otd = params.get('base_otd', 58.3)
    
    new_delay = max(0.5, base_delay - (pct * 6.2))
    new_cost = base_cost + (base_cost * pct * (cost_mult - 1.0) * 0.15)
    new_otd = min(98.0, base_otd + (pct * 33.0))
    
    return {
        "projected_otd_pct": run_monte_carlo(new_otd, 1.5),
        "projected_avg_delay_days": run_monte_carlo(new_delay, 0.4),
        "projected_total_cost": run_monte_carlo(new_cost, new_cost * 0.03)
    }

def dual_source_scenario(params):
    """What if we split orders across secondary vendors?"""
    split_pct = params.get('split_pct', 30) / 100.0
    sec_otd = params.get('secondary_otd', 90) / 100.0
    base_delay = params.get('base_delay', 7.5)
    base_cost = params.get('base_cost', 1034700.0)
    base_otd = params.get('base_otd', 58.3)
    
    new_otd = min(96.0, base_otd + (split_pct * sec_otd * 45.0))
    new_delay = max(1.0, base_delay - (split_pct * sec_otd * 7.0))
    new_cost = base_cost * (1.0 + (split_pct * 0.08))
    
    return {
        "projected_otd_pct": run_monte_carlo(new_otd, 1.2),
        "projected_avg_delay_days": run_monte_carlo(new_delay, 0.3),
        "projected_total_cost": run_monte_carlo(new_cost, new_cost * 0.02)
    }

def buffer_stock_scenario(params):
    """What if we increase safety buffer stock?"""
    buffer_days = params.get('buffer_days', 7)
    holding_cost_pct = params.get('holding_cost_pct', 20) / 100.0
    base_risk = params.get('production_stop_risk', 0.45)
    base_holding = params.get('base_holding_cost', 185000.0)
    
    new_risk = max(0.02, base_risk - (buffer_days * 0.035))
    new_holding = base_holding + ((buffer_days / 365.0) * holding_cost_pct * 360000.0)
    
    return {
        "projected_stop_risk_pct": run_monte_carlo(new_risk * 100.0, 1.0),
        "projected_annual_holding_cost": run_monte_carlo(new_holding, new_holding * 0.04)
    }

def vendor_consolidation_scenario(params):
    """What if we consolidate volume into top-tier strategic vendors?"""
    consolidation_pct = params.get('consolidation_pct', 25) / 100.0
    base_cost = params.get('base_cost', 1034700.0)
    base_otd = params.get('base_otd', 58.3)
    
    new_cost = base_cost * (1.0 - (consolidation_pct * 0.12))
    new_otd = min(95.0, base_otd + (consolidation_pct * 25.0))
    
    return {
        "projected_otd_pct": run_monte_carlo(new_otd, 1.5),
        "projected_total_cost": run_monte_carlo(new_cost, new_cost * 0.02)
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameterized Scenario Execution

# COMMAND ----------

# Safely extract widget parameters or use defaults
try:
    dbutils.widgets.text("scenario", "carrier_switch")
    dbutils.widgets.text("params_json", '{"pct_rerouted": 50, "cost_multiplier": 3.0, "base_delay": 7.5, "base_cost": 1034700, "base_otd": 58.3}')
    
    scenario_name = dbutils.widgets.get("scenario")
    params = json.loads(dbutils.widgets.get("params_json"))
except Exception:
    scenario_name = "carrier_switch"
    params = {"pct_rerouted": 50, "cost_multiplier": 3.0, "base_delay": 7.5, "base_cost": 1034700, "base_otd": 58.3}

logger.info(f"Running Scenario: '{scenario_name}' with parameters: {params}")

# Execute Selected Model
if scenario_name == "carrier_switch":
    simulation_output = carrier_switch_scenario(params)
elif scenario_name == "dual_source":
    simulation_output = dual_source_scenario(params)
elif scenario_name == "buffer_stock":
    simulation_output = buffer_stock_scenario(params)
elif scenario_name == "vendor_consolidation":
    simulation_output = vendor_consolidation_scenario(params)
else:
    simulation_output = carrier_switch_scenario(params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results to Simulation Delta Table

# COMMAND ----------

output_table = f"{CATALOG}.{SCHEMA}.simulation_results"

result_record = [{
    "simulation_id": f"SIM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
    "scenario_name": scenario_name,
    "input_parameters": json.dumps(params),
    "simulation_results": json.dumps(simulation_output),
    "executed_at": datetime.utcnow()
}]

results_schema = StructType([
    StructField("simulation_id", StringType(), False),
    StructField("scenario_name", StringType(), False),
    StructField("input_parameters", StringType(), False),
    StructField("simulation_results", StringType(), False),
    StructField("executed_at", TimestampType(), False)
])

results_df = spark.createDataFrame(result_record, schema=results_schema)

results_df.write.format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .saveAsTable(output_table)

logger.info(f"Simulation successfully completed and logged to {output_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulation Output Preview

# COMMAND ----------

print("\n" + "="*60)
print(f"SCENARIO PROJECTION: {scenario_name.upper()}")
print("="*60)
for metric, stats in simulation_output.items():
    print(f"• {metric:<30} -> Mean: {stats['mean']} (90% CI: {stats['p5']} to {stats['p95']})")
print("="*60 + "\n")

display(spark.table(output_table).orderBy(F.col("executed_at").desc()).limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Standalone API Function for Sub-Second Simulation Calls

# COMMAND ----------

def run_simulation_api(scenario_name, params):
    """
    Direct callable function for API endpoints / UI integration.
    Executes in < 5ms without Spark DataFrame overhead.
    """
    if scenario_name == "carrier_switch":
        return carrier_switch_scenario(params)
    elif scenario_name == "dual_source":
        return dual_source_scenario(params)
    elif scenario_name == "buffer_stock":
        return buffer_stock_scenario(params)
    elif scenario_name == "vendor_consolidation":
        return vendor_consolidation_scenario(params)
    elif scenario_name == "route_optimization":
        return route_optimization_scenario(params)
    else:
        return {"error": f"Unknown scenario: {scenario_name}"}

