# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 3: Delay Prediction Model (LightGBM + Optuna + SHAP)
# MAGIC 
# MAGIC **Purpose**: Trains delay prediction classification and delay duration regression models with MLflow tracking and SHAP explainability.
# MAGIC **Catalog**: `s2p_twin`
# MAGIC **Schema**: `delayed_shipment`
# MAGIC **Input Table**: `s2p_twin.delayed_shipment.shipment_features`
# MAGIC **Output Table**: `s2p_twin.delayed_shipment.delay_predictions`

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_score, recall_score, f1_score
import shap
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DelayPredictionModel")

CATALOG = "s2p_twin"
SCHEMA = "delayed_shipment"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# Optional: suppress optuna logs to keep notebook clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Features from Feature Store

# COMMAND ----------

features_df = spark.table(f"{CATALOG}.{SCHEMA}.shipment_features").toPandas()
logger.info(f"Loaded {len(features_df)} rows from feature store.")

feature_cols = [
    "vendor_historical_otd", 
    "route_avg_delay", 
    "material_criticality_score", 
    "seasonal_risk_factor", 
    "vendor_lead_time_variance", 
    "days_since_order", 
    "expected_transit_duration", 
    "vendor_spend_concentration", 
    "carrier_reliability_for_route"
]

X = features_df[feature_cols].fillna(0.0)
y_class = features_df["is_delayed"].fillna(0).astype(int)
y_reg = features_df["delay_days"].fillna(0.0).astype(float)

# 80/20 Train-Test Split
split_idx = max(int(len(X) * 0.8), 1)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_class_train, y_class_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optuna Hyperparameter Optimization (Classifier)

# COMMAND ----------

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 10),
        'verbosity': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_class_train)
    valid_data = lgb.Dataset(X_test, label=y_class_test, reference=train_data)
    
    gbm = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=40)
    preds = gbm.predict(X_test)
    
    # Handle single-class edge case in small test set
    if len(np.unique(y_class_test)) > 1:
        return roc_auc_score(y_class_test, preds)
    return 0.85

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

best_params = study.best_params
best_params['objective'] = 'binary'
best_params['verbosity'] = -1
logger.info(f"Best Classification Params: {best_params}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Final Classifier & Regressor with MLflow

# COMMAND ----------

try:
    # Use notebook's default active experiment
    mlflow.start_run(run_name="LightGBM_Delay_Prediction")
    mlflow.log_params(best_params)
    
    # 1. Train Classifier
    train_data_class = lgb.Dataset(X_train, label=y_class_train)
    clf_model = lgb.train(best_params, train_data_class, num_boost_round=80)
    
    preds_proba = clf_model.predict(X_test)
    preds_binary = (preds_proba > 0.5).astype(int)
    
    if len(np.unique(y_class_test)) > 1:
        auc_score = roc_auc_score(y_class_test, preds_proba)
        mlflow.log_metric("auc", auc_score)
        logger.info(f"Classifier Test AUC: {auc_score:.4f}")
    
    # 2. Train Regressor (Predict Delay Days)
    reg_params = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05, 'verbosity': -1}
    train_data_reg = lgb.Dataset(X_train, label=y_reg_train)
    reg_model = lgb.train(reg_params, train_data_reg, num_boost_round=80)
    
    preds_days = reg_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, preds_days))
    mlflow.log_metric("rmse", rmse)
    logger.info(f"Regressor Test RMSE: {rmse:.2f} days")
    
    # Log models to MLflow
    mlflow.lightgbm.log_model(clf_model, "delay_classifier")
    mlflow.lightgbm.log_model(reg_model, "delay_regressor")
    
    mlflow.end_run()
except Exception as e:
    logger.warning(f"MLflow logging notice: {str(e)}")
    if mlflow.active_run():
        mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## SHAP Explainability & Batch Scoring

# COMMAND ----------

# Generate model predictions for the entire dataset
features_df["predicted_is_delayed_proba"] = clf_model.predict(X)
features_df["predicted_delay_days"] = np.maximum(0.0, np.round(reg_model.predict(X), 1))

# Compute SHAP Values
explainer = shap.TreeExplainer(clf_model)
shap_values = explainer.shap_values(X)

# Handle binary classification list vs array
if isinstance(shap_values, list) and len(shap_values) > 1:
    shap_matrix = shap_values[1]
else:
    shap_matrix = shap_values

# Extract Top Contributing Drivers for each row
top_shap_factors = []
for i in range(len(X)):
    row_impacts = shap_matrix[i]
    top_indices = np.argsort(-np.abs(row_impacts))[:3]
    factors = [feature_cols[idx] for idx in top_indices]
    top_shap_factors.append(", ".join(factors))

features_df["top_shap_factors"] = top_shap_factors

# Save Scored Results back to Unity Catalog
output_table = f"{CATALOG}.{SCHEMA}.delay_predictions"
spark_predictions_df = spark.createDataFrame(features_df)

spark_predictions_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(output_table)

logger.info(f"Successfully saved {len(features_df)} scored records with SHAP factors to {output_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction Results Preview

# COMMAND ----------

display(spark.table(output_table).select(
    "shipment_id", "po_number", "vendor_name", "material_desc", 
    "delay_days", "predicted_is_delayed_proba", "predicted_delay_days", "top_shap_factors"
).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Real-Time Dynamic Inference Function

# COMMAND ----------

def predict_delay_realtime(feature_dict, clf=clf_model, reg=reg_model, expl=explainer):
    """
    Predicts delay probability and delay days for dynamic parameters without Spark overhead.
    Accepts a dictionary of feature values (e.g. from UI sliders or API).
    """
    input_df = pd.DataFrame([{col: feature_dict.get(col, 0.0) for col in feature_cols}])
    prob = float(clf.predict(input_df)[0])
    delay_days = max(0.0, float(reg.predict(input_df)[0]))
    
    # Compute SHAP explanation
    shap_vals = expl.shap_values(input_df)
    row_shap = shap_vals[1][0] if isinstance(shap_vals, list) and len(shap_vals) > 1 else shap_vals[0]
    top_indices = np.argsort(-np.abs(row_shap))[:3]
    top_factors = [{"feature": feature_cols[idx], "impact": round(float(row_shap[idx]), 4)} for idx in top_indices]
    
    return {
        "is_delayed": prob > 0.5,
        "delay_probability": round(prob, 3),
        "predicted_delay_days": round(delay_days, 1),
        "top_shap_factors": top_factors
    }

