# data-science-copilot

An agentic, low-dependency interactive sandbox application designed to perform multi-step data science analysis directly over raw text tables and custom CSV/JSON file datasets. The engine utilizes an autonomous code-generation and self-correction loop powered by Google Gemini to transform conversational prompts into clean visualization graphics.

---

## 🚀 Key Architectural Strengths

*   **Zero Third-Party Dependencies:** Written completely using Python's native Standard Library modules (`http.server`, `urllib`, `subprocess`, etc.). No overhead installation of `Flask`, `FastAPI`, `Pandas`, or `Numpy` is required.
*   **Self-Correcting RAG Feedback Loop:** If an AI-generated runtime script throws a `KeyError` or code exception inside the execution sandbox, the internal engine captures the execution context error and feeds it back into the prompt array for up to 3 automatic troubleshooting passes.
*   **Secure API Architecture:** Configured to strictly inject API authentications via standard HTTPS headers and cloud environment variables, keeping keys fully secure and out of the public git commit history.
*   **Native Chart Rendering Engine:** Outputs beautiful, fully interactive dark-themed SVG visualization blocks directly from standard JSON dictionary structures.

---

## 🛠️ Secure Setup & Local Execution

To preserve compliance with Google's public cloud parameters and prevent accidental security deactivations, this system reads values directly from your operating system shell.

### 1. Set Up Your Environment Variable

#### On Windows (Command Prompt / CMD):
```cmd
set GEMINI_API_KEY="AIzaSyYourSecretKeyHere"
