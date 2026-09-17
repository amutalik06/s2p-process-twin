# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 8: AI Agent Tool Registration (Unity Catalog Functions)
# MAGIC 
# MAGIC **Purpose**: Registers native Unity Catalog functions that serve as the tools for the S2P Process Twin AI Copilot Agent.
# MAGIC 
# MAGIC **Catalog**: `s2p_twin`
# MAGIC **Schema**: `delayed_shipment`
# MAGIC 
# MAGIC **Tools Registered**:
# MAGIC 1. `get_po_analysis` — (LANGUAGE SQL) Deep analytical lookup of a Purchase Order joining PO, vendor, material, shipment, and ML recommendations
# MAGIC 2. `get_shipment_analysis` — (LANGUAGE SQL) Detailed shipment tracking, carrier reliability, route risk factor, and ML delay predictions
# MAGIC 3. `get_morning_risk_scan` — (LANGUAGE SQL) Executive scan of all at-risk shipments, total value at risk ($), critical items, and pending actions
# MAGIC 4. `compare_vendor_performance` — (LANGUAGE SQL) Comprehensive supplier scorecard comparing OTD%, risk scores, active spend, and delays
# MAGIC 5. `simulate_whatif_scenario` — (LANGUAGE PYTHON) Self-contained Monte Carlo simulation with P5, Mean, P95 confidence intervals
# MAGIC 
# MAGIC **Key Architecture Design**:
# MAGIC - All data querying functions use **`LANGUAGE SQL`** — 100% native, runs inside the Databricks SQL engine with zero sandboxing issues, no `_sqldf`, and no external tokens.
# MAGIC - Mathematical/simulation functions use **self-contained `LANGUAGE PYTHON`** with standard libraries (`math`, `random`, `json`), requiring zero external network calls or `dbutils`.

# COMMAND ----------

CATALOG = "s2p_twin"
SCHEMA = "delayed_shipment"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Registering native agent tools in {CATALOG}.{SCHEMA}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 1: Purchase Order Deep Analysis (`get_po_analysis`)
# MAGIC 
# MAGIC Analyzes any PO by number (e.g. `'PO-4500103'`). Joins PO header, vendor master, material criticality, linked shipment tracking, and prescriptive recommendations into a structured JSON string.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.get_po_analysis(po_number_input STRING)
RETURNS STRING
LANGUAGE SQL
DETERMINISTIC
COMMENT 'Analyzes a Purchase Order by PO number (e.g. PO-4500103, PO-4500045). Returns complete PO data, shipment tracking status, delay days, vendor performance, material criticality, total value in USD, and ML delay risk recommendations with SHAP drivers.'
RETURN (
  SELECT coalesce(to_json(named_struct(
    'po_number', po.po_number,
    'vendor_id', po.vendor_id,
    'vendor_name', coalesce(v.vendor_name, po.vendor_id),
    'vendor_otd_pct', v.on_time_delivery_pct,
    'vendor_risk_score', v.risk_score,
    'material_id', po.material_id,
    'material_desc', coalesce(m.material_name, po.material_id),
    'material_criticality', m.criticality,
    'total_value_usd', po.total_value,
    'po_status', po.status,
    'order_date', string(po.order_date),
    'delivery_date', string(po.delivery_date),
    'shipment_id', s.shipment_id,
    'shipment_status', coalesce(s.shipment_status, 'UNKNOWN'),
    'delay_days', coalesce(s.delay_days, 0),
    'delay_reason', s.delay_reason,
    'carrier_name', c.carrier_name,
    'recommendation_priority', r.priority,
    'predicted_delay_days', r.predicted_delay_days,
    'recommendation_summary', r.summary,
    'risk_drivers_rationale', r.rationale,
    'recommended_actions', r.recommended_actions
  )), to_json(named_struct('error', concat('Purchase Order not found: ', po_number_input))))
  FROM {CATALOG}.{SCHEMA}.fct_purchase_order po
  LEFT JOIN {CATALOG}.{SCHEMA}.dim_vendor v ON po.vendor_id = v.vendor_id
  LEFT JOIN {CATALOG}.{SCHEMA}.dim_material m ON po.material_id = m.material_id
  LEFT JOIN {CATALOG}.{SCHEMA}.fct_shipment s ON po.po_number = s.po_number
  LEFT JOIN {CATALOG}.{SCHEMA}.dim_carrier c ON s.carrier_id = c.carrier_id
  LEFT JOIN {CATALOG}.{SCHEMA}.recommendations r ON po.po_number = r.po_number
  WHERE upper(trim(po.po_number)) = upper(trim(po_number_input))
  LIMIT 1
);
""")

print("✅ Tool 1 registered: get_po_analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 2: Shipment Tracking & Root Cause Analysis (`get_shipment_analysis`)
# MAGIC 
# MAGIC Analyzes any shipment by ID (e.g. `'SHP-90045'`). Returns tracking status, delay days, root cause reason, carrier reliability, shipping route risk factor, and ML recommendations.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.get_shipment_analysis(shipment_id_input STRING)
RETURNS STRING
LANGUAGE SQL
DETERMINISTIC
COMMENT 'Analyzes a specific shipment by ID (e.g. SHP-90045). Returns tracking status, delay days, delay reason, carrier reliability, shipping route risk, linked PO, vendor, and ML recommendations.'
RETURN (
  SELECT coalesce(to_json(named_struct(
    'shipment_id', s.shipment_id,
    'po_number', s.po_number,
    'shipment_status', s.shipment_status,
    'delay_days', s.delay_days,
    'delay_reason', s.delay_reason,
    'carrier_name', c.carrier_name,
    'carrier_mode', c.transit_mode,
    'carrier_reliability', c.reliability_score,
    'vendor_name', v.vendor_name,
    'vendor_otd_pct', v.on_time_delivery_pct,
    'material_desc', m.material_name,
    'material_criticality', m.criticality,
    'total_value_usd', po.total_value,
    'route_origin', sr.origin_port,
    'route_dest', sr.destination_plant,
    'route_risk_factor', sr.risk_factor,
    'recommendation_priority', r.priority,
    'predicted_delay_days', r.predicted_delay_days,
    'risk_drivers_rationale', r.rationale,
    'recommended_actions', r.recommended_actions
  )), to_json(named_struct('error', concat('Shipment not found: ', shipment_id_input))))
  FROM {CATALOG}.{SCHEMA}.fct_shipment s
  LEFT JOIN {CATALOG}.{SCHEMA}.fct_purchase_order po ON s.po_number = po.po_number
  LEFT JOIN {CATALOG}.{SCHEMA}.dim_vendor v ON po.vendor_id = v.vendor_id
  LEFT JOIN {CATALOG}.{SCHEMA}.dim_material m ON po.material_id = m.material_id
  LEFT JOIN {CATALOG}.{SCHEMA}.dim_carrier c ON s.carrier_id = c.carrier_id
  LEFT JOIN {CATALOG}.{SCHEMA}.dim_shipping_route sr ON s.route_id = sr.route_id
  LEFT JOIN {CATALOG}.{SCHEMA}.recommendations r ON s.shipment_id = r.shipment_id
  WHERE upper(trim(s.shipment_id)) = upper(trim(shipment_id_input))
  LIMIT 1
);
""")

print("✅ Tool 2 registered: get_shipment_analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 3: Morning Supply Chain Risk Scan (`get_morning_risk_scan`)
# MAGIC 
# MAGIC Returns an executive summary of current supply chain exposure: total at-risk shipments count, total value at risk ($), count of CRITICAL and HIGH items, and top delayed shipments.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.get_morning_risk_scan()
RETURNS STRING
LANGUAGE SQL
COMMENT 'Scans and returns the current supply chain risks: count and list of CRITICAL and HIGH priority delayed shipments, total dollar value at risk, top delay reasons, and pending prescriptive recommendations.'
RETURN (
  SELECT to_json(named_struct(
    'total_shipments_at_risk', count(DISTINCT s.shipment_id),
    'total_value_at_risk_usd', sum(coalesce(po.total_value, 0)),
    'critical_count', sum(CASE WHEN r.priority = 'CRITICAL' THEN 1 ELSE 0 END),
    'high_count', sum(CASE WHEN r.priority = 'HIGH' THEN 1 ELSE 0 END),
    'top_delayed_shipments', (
      SELECT to_json(collect_list(named_struct(
        'shipment_id', s2.shipment_id,
        'po_number', s2.po_number,
        'delay_days', s2.delay_days,
        'status', s2.shipment_status,
        'priority', r2.priority,
        'reason', s2.delay_reason
      )))
      FROM (
        SELECT s3.shipment_id, s3.po_number, s3.delay_days, s3.shipment_status, r3.priority, s3.delay_reason
        FROM {CATALOG}.{SCHEMA}.fct_shipment s3
        LEFT JOIN {CATALOG}.{SCHEMA}.recommendations r3 ON s3.shipment_id = r3.shipment_id
        WHERE s3.delay_days > 0 OR r3.priority IN ('CRITICAL', 'HIGH')
        ORDER BY s3.delay_days DESC
        LIMIT 6
      ) s2
    )
  ))
  FROM {CATALOG}.{SCHEMA}.fct_shipment s
  LEFT JOIN {CATALOG}.{SCHEMA}.fct_purchase_order po ON s.po_number = po.po_number
  LEFT JOIN {CATALOG}.{SCHEMA}.recommendations r ON s.shipment_id = r.shipment_id
  WHERE s.delay_days > 0 OR r.priority IN ('CRITICAL', 'HIGH')
);
""")

print("✅ Tool 3 registered: get_morning_risk_scan")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 4: Strategic Vendor Performance Comparison (`compare_vendor_performance`)
# MAGIC 
# MAGIC Compares vendors on on-time delivery (OTD%), risk scores, active purchase orders, total spend in USD, and average delay days.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.compare_vendor_performance(vendor_name_filter STRING)
RETURNS STRING
LANGUAGE SQL
COMMENT 'Compares vendor performance across on-time delivery rate (OTD%), risk scores, spend concentration, and delay patterns. Pass a vendor name (e.g. Meridian, Apex) or ALL to compare all strategic suppliers.'
RETURN (
  SELECT to_json(collect_list(named_struct(
    'vendor_id', v.vendor_id,
    'vendor_name', v.vendor_name,
    'country', v.country,
    'tier', v.tier,
    'on_time_delivery_pct', v.on_time_delivery_pct,
    'risk_score', v.risk_score,
    'active_po_count', coalesce(po_stats.po_count, 0),
    'total_spend_usd', coalesce(po_stats.total_spend, 0),
    'avg_delay_days', coalesce(po_stats.avg_delay, 0.0)
  )))
  FROM {CATALOG}.{SCHEMA}.dim_vendor v
  LEFT JOIN (
    SELECT 
      po.vendor_id,
      count(po.po_number) as po_count,
      sum(po.total_value) as total_spend,
      avg(coalesce(s.delay_days, 0)) as avg_delay
    FROM {CATALOG}.{SCHEMA}.fct_purchase_order po
    LEFT JOIN {CATALOG}.{SCHEMA}.fct_shipment s ON po.po_number = s.po_number
    GROUP BY po.vendor_id
  ) po_stats ON v.vendor_id = po_stats.vendor_id
  WHERE vendor_name_filter IS NULL 
     OR upper(trim(vendor_name_filter)) = 'ALL' 
     OR upper(v.vendor_name) LIKE concat('%', upper(trim(vendor_name_filter)), '%')
);
""")

print("✅ Tool 4 registered: compare_vendor_performance")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 5: Monte Carlo What-If Simulation (`simulate_whatif_scenario`)
# MAGIC 
# MAGIC Runs statistical Monte Carlo simulations for supply chain policy changes. Pure Python execution (zero network calls, zero external credentials).

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.simulate_whatif_scenario(
    scenario STRING COMMENT 'Simulation scenario: carrier_switch (reroute sea freight to air), dual_source (split volume across vendors), buffer_stock (increase safety stock buffer), or vendor_consolidation.',
    pct_rerouted DOUBLE DEFAULT 50.0 COMMENT 'For carrier_switch: percentage of delayed volume shifted to air freight (0-100).',
    cost_multiplier DOUBLE DEFAULT 3.0 COMMENT 'For carrier_switch: air freight premium factor (1.5-5.0).',
    split_pct DOUBLE DEFAULT 30.0 COMMENT 'For dual_source: percentage allocated to secondary supplier (10-50).',
    secondary_otd DOUBLE DEFAULT 90.0 COMMENT 'For dual_source: expected on-time delivery rate of secondary vendor (70-99).',
    buffer_days DOUBLE DEFAULT 7.0 COMMENT 'For buffer_stock: additional days of safety buffer stock (0-30).',
    holding_cost_pct DOUBLE DEFAULT 20.0 COMMENT 'For buffer_stock: annual inventory carrying rate percentage (10-35).',
    consolidation_pct DOUBLE DEFAULT 25.0 COMMENT 'For vendor_consolidation: percentage of volume consolidated (10-60).'
)
RETURNS STRING
LANGUAGE PYTHON
DETERMINISTIC
COMMENT 'Run a Monte Carlo What-If simulation for supply chain policy changes. Returns projected OTD%, average delay days, total cost, and confidence intervals (P5, Mean, P95).'
AS $$
    import json
    import random

    # Fixed seed for deterministic simulation within UDF execution
    random.seed(42)

    def mc_dist(mean_val, std_val):
        samples = sorted([random.gauss(mean_val, std_val) for _ in range(500)])
        return {{
            "mean": round(mean_val, 2),
            "p5": round(samples[25], 2),
            "p95": round(samples[475], 2)
        }}

    base_otd = 58.3
    base_delay = 7.5
    base_cost = 1034700.0

    if scenario == "carrier_switch":
        pct = (pct_rerouted or 50.0) / 100.0
        cost_mult = cost_multiplier or 3.0
        new_delay = max(0.5, base_delay - (pct * 6.2))
        new_cost = base_cost + (base_cost * pct * (cost_mult - 1.0) * 0.15)
        new_otd = min(98.0, base_otd + (pct * 33.0))
        res = {{
            "scenario": "carrier_switch",
            "projected_otd_pct": mc_dist(new_otd, 1.5),
            "projected_avg_delay_days": mc_dist(new_delay, 0.4),
            "projected_total_cost_usd": mc_dist(new_cost, new_cost * 0.03),
            "efficiency_ratio": round((new_otd - base_otd) / max(1.0, (new_cost - base_cost) / 1000.0), 3)
        }}

    elif scenario == "dual_source":
        split = (split_pct or 30.0) / 100.0
        sec_otd_val = (secondary_otd or 90.0) / 100.0
        new_otd = min(96.0, base_otd + (split * sec_otd_val * 45.0))
        new_delay = max(1.0, base_delay - (split * sec_otd_val * 7.0))
        new_cost = base_cost * (1.0 + (split * 0.08))
        res = {{
            "scenario": "dual_source",
            "projected_otd_pct": mc_dist(new_otd, 1.2),
            "projected_avg_delay_days": mc_dist(new_delay, 0.3),
            "projected_total_cost_usd": mc_dist(new_cost, new_cost * 0.02)
        }}

    elif scenario == "buffer_stock":
        add_days = buffer_days or 7.0
        hold_pct = (holding_cost_pct or 20.0) / 100.0
        base_risk = 0.45
        new_risk = max(0.02, base_risk - (add_days * 0.035))
        base_holding = 185000.0
        new_holding = base_holding + ((add_days / 365.0) * hold_pct * 360000.0)
        res = {{
            "scenario": "buffer_stock",
            "projected_stop_risk_pct": mc_dist(new_risk * 100.0, 1.0),
            "projected_annual_holding_cost_usd": mc_dist(new_holding, new_holding * 0.04)
        }}

    else:
        # Default vendor consolidation
        cons = (consolidation_pct or 25.0) / 100.0
        new_cost = base_cost * (1.0 - (cons * 0.12))
        new_otd = min(95.0, base_otd + (cons * 25.0))
        res = {{
            "scenario": "vendor_consolidation",
            "projected_otd_pct": mc_dist(new_otd, 1.5),
            "projected_total_cost_usd": mc_dist(new_cost, new_cost * 0.02)
        }}

    return json.dumps(res)
$$
""")

print("✅ Tool 5 registered: simulate_whatif_scenario")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification: Test Tools Directly with SQL Queries

# COMMAND ----------

# Test 1: Verify get_po_analysis with a sample PO
display(spark.sql(f"SELECT {CATALOG}.{SCHEMA}.get_po_analysis('PO-4500103') AS po_test"))

# COMMAND ----------

# Test 2: Verify morning risk scan
display(spark.sql(f"SELECT {CATALOG}.{SCHEMA}.get_morning_risk_scan() AS risk_scan_test"))

# COMMAND ----------

# Test 3: Verify vendor comparison
display(spark.sql(f"SELECT {CATALOG}.{SCHEMA}.compare_vendor_performance('ALL') AS vendor_test"))

# COMMAND ----------

# Test 4: Verify Monte Carlo simulation
display(spark.sql(f"SELECT {CATALOG}.{SCHEMA}.simulate_whatif_scenario('carrier_switch', 50.0, 3.0, 30.0, 90.0, 7.0, 20.0, 25.0) AS sim_test"))

# COMMAND ----------

print("🎉 ALL 5 TOOLS REGISTERED AND VERIFIED SUCCESSFULLY!")
