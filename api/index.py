"""
S2P Live Process Twin — Vercel Serverless CORS Proxy
=====================================================
Serverless Flask proxy running on Vercel to forward requests
from the SAP Fiori Process Twin UI to Databricks REST APIs.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VercelBDCProxy")

# Optional environment variable fallbacks (can be set in Vercel Project Settings)
ENV_WORKSPACE = os.environ.get("DATABRICKS_WORKSPACE_URL", "https://343749907984660.0.gcp.databricks.com").rstrip("/")
ENV_WAREHOUSE = os.environ.get("DATABRICKS_WAREHOUSE_ID", "f0d413f0dd7ad7ac")
ENV_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
ENV_ENDPOINT = os.environ.get("DATABRICKS_ENDPOINT_NAME", "s2p-twin-engine")

# ──────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "S2P Twin Vercel Serverless CORS Proxy",
        "version": "2.4",
        "env_configured": bool(ENV_TOKEN)
    })


# ──────────────────────────────────────────────────────────────
# Connection Test
# ──────────────────────────────────────────────────────────────

@app.route("/proxy/test", methods=["POST"])
def proxy_test():
    try:
        body = request.get_json(force=True) if request.data else {}
        workspace_url = (body.get("workspaceUrl") or ENV_WORKSPACE).rstrip("/")
        token = body.get("token") or ENV_TOKEN
        warehouse_id = body.get("warehouseId") or ENV_WAREHOUSE

        if not workspace_url or not token:
            return jsonify({"success": False, "message": "Missing workspace URL or Personal Access Token"}), 400

        resp = requests.post(
            f"{workspace_url}/api/2.0/sql/statements",
            json={
                "statement": "SELECT 1 AS connection_test",
                "warehouse_id": warehouse_id,
                "wait_timeout": "10s"
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            timeout=15
        )

        data = resp.json()
        state = data.get("status", {}).get("state", "UNKNOWN")

        if state == "SUCCEEDED":
            return jsonify({"success": True, "state": state, "message": "Connection to SAP Databricks successful!"})
        else:
            error_msg = data.get("status", {}).get("error", {}).get("message", "Unknown error")
            return jsonify({"success": False, "state": state, "message": error_msg}), 400

    except requests.exceptions.ConnectionError as e:
        return jsonify({"success": False, "message": f"Cannot reach workspace URL. Verify URL. Error: {str(e)}"}), 502
    except requests.exceptions.Timeout:
        return jsonify({"success": False, "message": "Connection test timed out (15s)"}), 504
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ──────────────────────────────────────────────────────────────
# SQL Statement Execution API Proxy
# ──────────────────────────────────────────────────────────────

@app.route("/proxy/sql", methods=["POST"])
def proxy_sql():
    try:
        body = request.get_json(force=True)
        workspace_url = (body.get("workspaceUrl") or ENV_WORKSPACE).rstrip("/")
        token = body.get("token") or ENV_TOKEN
        warehouse_id = body.get("warehouseId") or ENV_WAREHOUSE
        statement = body.get("statement")

        if not statement:
            return jsonify({"error": "Missing 'statement' parameter"}), 400

        databricks_url = f"{workspace_url}/api/2.0/sql/statements"
        databricks_body = {
            "statement": statement,
            "warehouse_id": warehouse_id,
            "wait_timeout": "30s"
        }

        resp = requests.post(
            databricks_url,
            json=databricks_body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            timeout=60
        )

        return jsonify(resp.json()), resp.status_code

    except KeyError as e:
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except requests.exceptions.ConnectionError as e:
        return jsonify({"error": f"Cannot reach Databricks workspace: {str(e)}"}), 502
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request to Databricks timed out (60s)"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────
# SQL Statement Status Polling
# ──────────────────────────────────────────────────────────────

@app.route("/proxy/statement/<statement_id>", methods=["POST", "GET"])
def proxy_statement(statement_id):
    try:
        body = request.get_json(force=True) if request.data else {}
        workspace_url = (body.get("workspaceUrl") or ENV_WORKSPACE).rstrip("/")
        token = body.get("token") or ENV_TOKEN

        databricks_url = f"{workspace_url}/api/2.0/sql/statements/{statement_id}"
        resp = requests.get(
            databricks_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────
# Model Serving Invocations Proxy
# ──────────────────────────────────────────────────────────────

@app.route("/proxy/serving", methods=["POST"])
def proxy_serving():
    try:
        body = request.get_json(force=True)
        workspace_url = (body.get("workspaceUrl") or ENV_WORKSPACE).rstrip("/")
        token = body.get("token") or ENV_TOKEN
        endpoint_name = body.get("endpointName") or ENV_ENDPOINT
        payload = body.get("payload")

        if not payload:
            return jsonify({"error": "Missing 'payload' parameter"}), 400

        databricks_url = f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"

        resp = requests.post(
            databricks_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            timeout=30
        )

        return jsonify(resp.json()), resp.status_code

    except KeyError as e:
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except requests.exceptions.ConnectionError as e:
        return jsonify({"error": f"Cannot reach Model Serving endpoint: {str(e)}"}), 502
    except requests.exceptions.Timeout:
        return jsonify({"error": "Model Serving request timed out (30s)"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────
# AI Agent Copilot Proxy (s2p-procurement-copilot)
# ──────────────────────────────────────────────────────────────

@app.route("/proxy/agent", methods=["POST"])
def proxy_agent():
    try:
        body = request.get_json(force=True)
        workspace_url = (body.get("workspaceUrl") or ENV_WORKSPACE).rstrip("/")
        token = body.get("token") or ENV_TOKEN
        endpoint_name = body.get("endpointName", "s2p-procurement-copilot")
        message = body.get("message")
        chat_history = body.get("chatHistory", [])

        if not message:
            return jsonify({"error": "Missing 'message' parameter"}), 400

        databricks_url = f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"

        # Databricks Model Serving for MLflow LangChain agents uses ChatModel schema:
        # {"messages": [{"role": "user", "content": "..."}, ...]}
        messages = []
        if chat_history:
            for msg in chat_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        payload = {"messages": messages}

        resp = requests.post(
            databricks_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            timeout=120
        )

        return jsonify(resp.json()), resp.status_code

    except KeyError as e:
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except requests.exceptions.ConnectionError as e:
        return jsonify({"error": f"Cannot reach Agent endpoint: {str(e)}"}), 502
    except requests.exceptions.Timeout:
        return jsonify({"error": "Agent request timed out (120s). The agent may be warming up — try again."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=3001, debug=True)