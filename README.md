# Civic_AI

## Setup (using uv)

1. **Install uv** (if you don’t have it):

   **Windows (PowerShell):**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   **Linux/macOS:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Go to the project and sync the environment:**
   ```bash
   cd Civic_AI
   uv sync
   ```

   This creates a virtual environment and installs all dependencies from `pyproject.toml` and `uv.lock`.

3. **Run Python or Jupyter:**
   ```bash
   uv run python main.py
   uv run jupyter notebook
   ```

   Or activate the venv and use Python directly:
   ```bash
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```