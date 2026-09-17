# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 6: Real-Time Serving Endpoint Deployment
# MAGIC 
# MAGIC **Purpose**: Packages the LightGBM Delay Predictor + Monte Carlo Simulation Engine into a single
# MAGIC Databricks **Model Serving Endpoint** for sub-second REST API responses from the Live Process Twin UI.
# MAGIC 
# MAGIC **Endpoint Name**: `s2p-twin-engine`
# MAGIC 
# MAGIC **Capabilities**:
# MAGIC - `predict` → Binary delay classification + delay days regression + SHAP top factors
# MAGIC - `simulate` → Monte Carlo what-if scenarios (carrier switch, dual source, buffer stock, vendor consolidation)
# MAGIC - `recommend` → Dynamic priority scoring with configurable thresholds

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ServingEndpointDeployment")

CATALOG = "s2p_twin"
SCHEMA = "delayed_shipment"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Unified Twin Engine PyFunc Model
# MAGIC 
# MAGIC This custom MLflow `pyfunc` model wraps ALL twin capabilities into a single serving endpoint.
# MAGIC The `action` field in the input determines which capability to invoke.

# COMMAND ----------

class S2PTwinEngine(mlflow.pyfunc.PythonModel):
    """
    Unified Process Twin Engine for Databricks Model Serving.
    
    Accepts JSON input with an 'action' field:
      - action="simulate"  → Monte Carlo what-if simulation
      - action="predict"   → Delay prediction with SHAP factors
      - action="recommend" → Dynamic recommendation priority scoring
      - action="kpis"      → Live KPI computation from feature store
    """
    
    def load_context(self, context):
        """Load trained LightGBM models from artifacts."""
        import lightgbm as lgb
        import shap
        
        self.classifier = lgb.Booster(model_file=context.artifacts["classifier_path"])
        self.regressor = lgb.Booster(model_file=context.artifacts["regressor_path"])
        self.explainer = shap.TreeExplainer(self.classifier)
        self.feature_cols = [
            "vendor_historical_otd", "route_avg_delay", "material_criticality_score",
            "seasonal_risk_factor", "vendor_lead_time_variance", "days_since_order",
            "expected_transit_duration", "vendor_spend_concentration", "carrier_reliability_for_route"
        ]
    
    def predict(self, context, model_input):
        """
        Dispatch to the appropriate action handler based on the 'action' field.
        """
        import pandas as pd
        
        results = []
        for _, row in model_input.iterrows():
            action = row.get("action", "simulate")
            params = json.loads(row.get("params", "{}")) if isinstance(row.get("params"), str) else row.get("params", {})
            
            if action == "simulate":
                result = self._simulate(params)
            elif action == "predict":
                result = self._predict_delay(params)
            elif action == "recommend":
                result = self._recommend(params)
            elif action == "kpis":
                result = self._compute_kpis(params)
            else:
                result = {"error": f"Unknown action: {action}"}
            
            results.append(json.dumps(result))
        
        return pd.DataFrame({"result": results})
    
    # ── Simulation Engine ──────────────────────────────────────
    
    def _run_monte_carlo(self, base_value, std_dev, iterations=1000):
        samples = np.random.normal(base_value, max(0.01, std_dev), iterations)
        return {
            "mean": round(float(np.mean(samples)), 2),
            "p5": round(float(np.percentile(samples, 5)), 2),
            "p95": round(float(np.percentile(samples, 95)), 2)
        }
    
    def _simulate(self, params):
        scenario = params.get("scenario", "carrier_switch")
        
        if scenario == "carrier_switch":
            pct = params.get("pct_rerouted", 50) / 100.0
            cost_mult = params.get("cost_multiplier", 3.0)
            base_delay = params.get("base_delay", 7.5)
            base_cost = params.get("base_cost", 1034700.0)
            base_otd = params.get("base_otd", 58.3)
            
            new_otd = min(98.0, base_otd + (pct * 33.0))
            new_delay = max(0.5, base_delay - (pct * 6.2))
            new_cost = base_cost + (base_cost * pct * (cost_mult - 1.0) * 0.15)
            at_risk = max(0, round(6 - (pct * 5)))
            
            return {
                "scenario": scenario,
                "otd_pct": self._run_monte_carlo(new_otd, 1.5),
                "avg_delay_days": self._run_monte_carlo(new_delay, 0.4),
                "total_cost": self._run_monte_carlo(new_cost, new_cost * 0.03),
                "at_risk_lines": at_risk
            }
        
        elif scenario == "dual_source":
            split = params.get("split_pct", 30) / 100.0
            sec_otd = params.get("secondary_otd", 90) / 100.0
            base_delay = params.get("base_delay", 7.5)
            base_cost = params.get("base_cost", 1034700.0)
            base_otd = params.get("base_otd", 58.3)
            
            new_otd = min(96.0, base_otd + (split * sec_otd * 45.0))
            new_delay = max(1.0, base_delay - (split * sec_otd * 7.0))
            new_cost = base_cost * (1.0 + (split * 0.08))
            risk_score = max(0.1, 0.62 - (split * 0.8))
            
            return {
                "scenario": scenario,
                "otd_pct": self._run_monte_carlo(new_otd, 1.2),
                "avg_delay_days": self._run_monte_carlo(new_delay, 0.3),
                "total_cost": self._run_monte_carlo(new_cost, new_cost * 0.02),
                "supply_risk_score": self._run_monte_carlo(risk_score, 0.03)
            }
        
        elif scenario == "buffer_stock":
            add_days = params.get("buffer_days", 7)
            hold_pct = params.get("holding_cost_pct", 20) / 100.0
            
            new_risk = max(0.02, 0.45 - (add_days * 0.035))
            additional_holding = (add_days / 365) * hold_pct * 120000 * 3
            new_holding = 185000 + additional_holding
            at_risk = max(0, round(6 - (add_days * 0.5)))
            
            return {
                "scenario": scenario,
                "production_stop_risk": self._run_monte_carlo(new_risk, 0.02),
                "avg_buffer_days": round(5.7 + add_days, 1),
                "holding_cost_annual": self._run_monte_carlo(new_holding, new_holding * 0.04),
                "at_risk_lines": at_risk
            }
        
        elif scenario == "vendor_consolidation":
            consolidation = params.get("consolidation_pct", 25) / 100.0
            base_cost = params.get("base_cost", 1034700.0)
            base_otd = params.get("base_otd", 58.3)
            
            new_cost = base_cost * (1.0 - (consolidation * 0.12))
            new_otd = min(95.0, base_otd + (consolidation * 25.0))
            
            return {
                "scenario": scenario,
                "otd_pct": self._run_monte_carlo(new_otd, 1.5),
                "total_cost": self._run_monte_carlo(new_cost, new_cost * 0.02)
            }
        
        return {"error": f"Unknown scenario: {scenario}"}
    
    # ── Delay Prediction ───────────────────────────────────────
    
    def _predict_delay(self, params):
        import pandas as pd
        
        features = {col: params.get(col, 0.0) for col in self.feature_cols}
        X = pd.DataFrame([features])
        
        delay_prob = float(self.classifier.predict(X)[0])
        delay_days = max(0.0, float(self.regressor.predict(X)[0]))
        
        # SHAP
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            row_shap = shap_values[1][0]
        else:
            row_shap = shap_values[0]
        
        top_indices = np.argsort(-np.abs(row_shap))[:3]
        top_factors = [
            {"feature": self.feature_cols[idx], "impact": round(float(row_shap[idx]), 4)}
            for idx in top_indices
        ]
        
        return {
            "delay_probability": round(delay_prob, 3),
            "predicted_delay_days": round(delay_days, 1),
            "top_shap_factors": top_factors,
            "is_delayed": delay_prob > 0.5
        }
    
    # ── Dynamic Recommendation ─────────────────────────────────
    
    def _recommend(self, params):
        prediction = self._predict_delay(params)
        delay_days = prediction["predicted_delay_days"]
        buffer = params.get("buffer_stock", 10)
        criticality = params.get("material_criticality_score", 2)
        
        crit_threshold = params.get("critical_threshold", 8)
        high_threshold = params.get("high_threshold", 5)
        
        if delay_days >= crit_threshold and buffer <= 5 and criticality >= 3:
            priority = "CRITICAL"
            actions = ["Emergency Air Expedite", "Alt Vendor Source", "Escalate to VP"]
        elif delay_days >= high_threshold or (delay_days >= 3 and buffer <= 3):
            priority = "HIGH"
            actions = ["Split Source", "Fast-Track Customs", "Reschedule Assembly"]
        elif delay_days >= 2:
            priority = "MEDIUM"
            actions = ["Contact Carrier", "Verify Buffer Stock"]
        else:
            priority = "LOW"
            actions = ["Standard Monitoring", "Acknowledge"]
        
        return {
            **prediction,
            "priority": priority,
            "recommended_actions": actions,
            "buffer_remaining": buffer - delay_days,
            "thresholds_used": {"critical": crit_threshold, "high": high_threshold}
        }
    
    # ── KPI Aggregation ────────────────────────────────────────
    
    def _compute_kpis(self, params):
        return {
            "note": "KPIs should be fetched via SQL API for live data.",
            "base_otd_pct": params.get("base_otd", 58.3),
            "base_avg_delay": params.get("base_delay", 7.5),
            "at_risk_count": params.get("at_risk_count", 6)
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Trained LightGBM Models as Artifacts & Log the PyFunc Model

# COMMAND ----------

import os
import tempfile

# Re-train lightweight models to save as Booster files
features_df = spark.table(f"{CATALOG}.{SCHEMA}.shipment_features").toPandas()

feature_cols = [
    "vendor_historical_otd", "route_avg_delay", "material_criticality_score",
    "seasonal_risk_factor", "vendor_lead_time_variance", "days_since_order",
    "expected_transit_duration", "vendor_spend_concentration", "carrier_reliability_for_route"
]

X = features_df[feature_cols].fillna(0.0)
y_class = features_df["is_delayed"].fillna(0).astype(int)
y_reg = features_df["delay_days"].fillna(0.0).astype(float)

import lightgbm as lgb

# Train classifier
clf_data = lgb.Dataset(X, label=y_class)
clf_params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'num_leaves': 31, 'learning_rate': 0.05}
clf_model = lgb.train(clf_params, clf_data, num_boost_round=80)

# Train regressor
reg_data = lgb.Dataset(X, label=y_reg)
reg_params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'learning_rate': 0.05}
reg_model = lgb.train(reg_params, reg_data, num_boost_round=80)

# Save to temp files
tmp_dir = tempfile.mkdtemp()
clf_path = os.path.join(tmp_dir, "classifier.txt")
reg_path = os.path.join(tmp_dir, "regressor.txt")
clf_model.save_model(clf_path)
reg_model.save_model(reg_path)

logger.info(f"Models saved to {tmp_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register as MLflow PyFunc Model in Unity Catalog

# COMMAND ----------

artifacts = {
    "classifier_path": clf_path,
    "regressor_path": reg_path
}

import sys

# Dynamically align container Python version with Serverless runtime (e.g. Python 3.11)
# to avoid cloudpickle bytecode deserialization mismatch (16 vs 18 arguments)
py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

conda_env = {
    "channels": ["defaults", "conda-forge"],
    "dependencies": [
        f"python={py_version}",
        "pip",
        {
            "pip": [
                f"mlflow=={mlflow.__version__}",
                "cloudpickle",
                "lightgbm",
                "shap",
                "numpy",
                "pandas"
            ]
        }
    ],
    "name": "s2p_twin_env"
}

registered_model_name = f"{CATALOG}.{SCHEMA}.s2p_twin_engine"

with mlflow.start_run(run_name="S2P_Twin_Engine_Serving"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="s2p_twin_engine",
        python_model=S2PTwinEngine(),
        artifacts=artifacts,
        conda_env=conda_env,
        registered_model_name=registered_model_name
    )
    logger.info(f"Model registered: {registered_model_name}")
    logger.info(f"Model URI: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy to Databricks Model Serving Endpoint
# MAGIC 
# MAGIC After registering the model above, create the serving endpoint via the Databricks UI:
# MAGIC 
# MAGIC 1. Go to **Serving** (left sidebar) → **Create serving endpoint**
# MAGIC 2. **Endpoint name**: `s2p-twin-engine`
# MAGIC 3. **Served entities**: Click **Select entity** → Choose `s2p_twin.delayed_shipment.s2p_twin_engine` → Select latest version
# MAGIC 4. **Compute type**: **CPU** (sufficient for our model)
# MAGIC 5. **Compute size**: **Small**
# MAGIC 6. Click **Create**
# MAGIC 7. Wait for status to show **Ready** (green checkmark)
# MAGIC 
# MAGIC ### Test the Endpoint
# MAGIC 
# MAGIC Use the **Query endpoint** panel on the right side of the serving page:
# MAGIC 
# MAGIC ```json
# MAGIC {
# MAGIC   "dataframe_records": [
# MAGIC     {
# MAGIC       "action": "simulate",
# MAGIC       "params": "{\"scenario\": \"carrier_switch\", \"pct_rerouted\": 60, \"base_delay\": 7.5, \"base_cost\": 1034700, \"base_otd\": 58.3, \"cost_multiplier\": 3.0}"
# MAGIC     }
# MAGIC   ]
# MAGIC }
# MAGIC ```
# MAGIC 
# MAGIC Expected response:
# MAGIC ```json
# MAGIC {
# MAGIC   "predictions": [
# MAGIC     {
# MAGIC       "result": "{\"scenario\": \"carrier_switch\", \"otd_pct\": {\"mean\": 78.1, \"p5\": 75.6, \"p95\": 80.5}, ...}"
# MAGIC     }
# MAGIC   ]
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Test via REST API (cURL)
# MAGIC 
# MAGIC ```bash
# MAGIC curl -X POST \
# MAGIC   https://<your-workspace>.cloud.databricks.com/serving-endpoints/s2p-twin-engine/invocations \
# MAGIC   -H "Authorization: Bearer <YOUR_DATABRICKS_PAT>" \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d '{
# MAGIC     "dataframe_records": [
# MAGIC       {
# MAGIC         "action": "simulate",
# MAGIC         "params": "{\"scenario\": \"carrier_switch\", \"pct_rerouted\": 75, \"base_delay\": 7.5, \"base_cost\": 1034700, \"base_otd\": 58.3, \"cost_multiplier\": 3.5}"
# MAGIC       }
# MAGIC     ]
# MAGIC   }'
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serving Endpoint Summary
# MAGIC 
# MAGIC | Property | Value |
# MAGIC |----------|-------|
# MAGIC | **Endpoint Name** | `s2p-twin-engine` |
# MAGIC | **Registered Model** | `s2p_twin.delayed_shipment.s2p_twin_engine` |
# MAGIC | **Capabilities** | `simulate`, `predict`, `recommend`, `kpis` |
# MAGIC | **Expected Latency** | < 100ms per request |
# MAGIC | **Authentication** | Databricks Personal Access Token (Bearer) |
# MAGIC | **REST URL** | `https://<workspace>/serving-endpoints/s2p-twin-engine/invocations` |
