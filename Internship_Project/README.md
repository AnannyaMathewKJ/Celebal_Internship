# 🚀 Data Science Co-Pilot (Celebal Technologies Internship Project)

An **Autonomous Data Science Co-Pilot** web application that transforms raw datasets (CSV/JSON) into actionable visual insights and data quality audits. Powered by an LLM-driven multi-step execution agent, it automatically generates Python code, runs it in an isolated sandbox, dynamically handles errors using a RAG (Retrieval-Augmented Generation) document lookup system, and visualizes results.

---

## 🌟 Key Features

* **🤖 Autonomous Python Code Generation:** Converts plain English analytical queries into executable Python code without relying on external heavy dependencies like `pandas` or `numpy`.
* **🛡️ Isolated Code Sandbox Execution:** Executes generated code in a secure local sandbox with stdout/stderr capture and strict timeout controls.
* **🔄 Self-Correction Engine with RAG Manuals:** If a generated script fails (e.g., `KeyError`, `ValueError`, `ModuleNotFoundError`), the system retrieves targeted RAG manuals and prompts the model to self-correct up to 3 attempts automatically.
* **📊 Dynamic SVG Chart Generation:** Built-in geometric SVG renderer that builds custom **Bar, Line, Area, Scatter, and Pie** charts on the fly without heavy frontend libraries.
* **💼 Preset Use Cases:** Includes pre-loaded datasets and standard queries (Sales Dashboards, Data Quality Audits, Trend Analysis, Cohort Analysis, and Ad-hoc Inventory Queries).
* **📁 Custom Dataset Support:** Multipart file upload for analyzing your own CSV or JSON files.

---

## 🏗️ Architecture & Workflow

1. **User Query & Dataset Ingestion:** The user selects a preset or uploads a CSV/JSON file and submits a natural language query.
2. **LLM Code Synthesis:** Hugging Face Inference API (`openai/gpt-oss-120b:cerebras` or configurable models) constructs standard Python code.
3. **Sandbox Execution:** Code is written to a temporary `sandbox.py` file and executed on the local runtime (`data_file` input).
4. **Error Interception & RAG Feedback:**
* **Success:** Parsed JSON output (chart specs + insight summary) is rendered instantly on the dashboard.
* **Failure:** Tracing intercepts the `stderr`, retrieves relevant troubleshooting documentation, and re-prompts the model for self-healing.



---

## 🛠️ Prerequisites & Installation

### 1. Prerequisites

* **Python 3.8+** installed on your system.
* A valid **Hugging Face User Access Token** (with Inference API permissions).

### 2. Repository Setup

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/your-username/data-science-copilot.git
cd data-science-copilot

```

### 3. Environment Configuration

Create a `.env` file in the root directory of the project:

```env
HF_API_KEY=your_huggingface_access_token_here
HF_MODEL=openai/gpt-oss-120b:cerebras

```

### 4. Optional Dependencies

Install `python-dotenv` if you prefer loading the `.env` automatically:

```bash
pip install python-dotenv

```

*(Note: The core server uses Python standard modules `http.server`, `socketserver`, `urllib`, `csv`, `json`, `subprocess`, etc.)*

---
