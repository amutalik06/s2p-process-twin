# Motiveminds S2P Live Process Twin

> **AI-Powered Prescriptive Analytics & Simulation on SAP Databricks**  
> Built with the SAP Fiori Horizon Design System (Morning Light & Evening Dark).

---

## 🌟 Architecture Overview

This project is deployed to **Vercel** with a dual-layer architecture:

1. **Frontend (`index.html`)**:
   - Single-page application using the **SAP Fiori Horizon Design System**.
   - 5 interactive tabs: Executive Dashboard, Semantic Data Explorer, Prescriptive Action Cockpit, Horizon 4 What-If Simulator, and Closed-Loop Audit Trail.
   - Dual-theme support: Morning Horizon (Light Mode) and Evening Horizon (Dark Mode).

2. **Serverless CORS Proxy (`api/index.py`)**:
   - High-performance Python serverless function running on Vercel.
   - Forwards browser requests to **SAP Databricks REST APIs**:
     - SQL Statement Execution API (`/api/2.0/sql/statements`)
     - Model Serving Endpoint (`/serving-endpoints/s2p-twin-engine/invocations`)
   - Bypasses browser CORS restrictions seamlessly without any local client software.

---

## 🚀 Live Deployment on Vercel

1. Import this repository into [Vercel](https://vercel.com).
2. Framework Preset: **Other**.
3. (Optional) In **Project Settings → Environment Variables**, configure:
   - `DATABRICKS_WORKSPACE_URL`: e.g. `https://343749907984660.0.gcp.databricks.com`
   - `DATABRICKS_WAREHOUSE_ID`: `f0d413f0dd7ad7ac`
   - `DATABRICKS_TOKEN`: `<your-personal-access-token>`
   - `DATABRICKS_ENDPOINT_NAME`: `s2p-twin-engine`
4. Click **Deploy**.

---

## 🏢 Corporate Branding
Designed & Developed by **Motiveminds Consulting Pvt Ltd** for the SAP Business Data Cloud & Databricks ecosystem.
