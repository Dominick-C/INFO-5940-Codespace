# How to Run
## 1) Create/activate a virtual environment

* If VS Code / Codespaces prompts to create a `.venv`, click Create
* Or do it yourself in the terminal:

  ```bash
  python -m venv .venv
  source .venv/bin/activate   # Linux/macOS/Codespaces
  # .venv\Scripts\activate    # Windows PowerShell
  ```
this step is not always needed and can be ignored, but my code spaces has been yelling at me to do this sometimes so im including this for proprietys sake. I talked to the ta and she said it shouldnt happen during grading, but this is here just exists

## 2) Install dependencies
Sometimes the code is yeling at me that we dodnt install streamlet. im not sue why, so the folllowing is included to assist here. in case it dosent run

  ```bash
  pip install streamlit
  ```

  If you hit a `ModuleNotFoundError` later, install that missing package the same way (e.g., `pip install name-of-package`) while your `.venv` is active.

## 3) Put your API keys in the code (for testing)

Open the main app file (e.g., `assign_2.py`). Look for lines that say **`put ____ key here`** and replace each blank with your actual key value (keep the quotes if the value is a string).
Examples you might see:

* `put openai key here`
* `put tavily key here`


## 4) Run the app

From the repo root:

```bash
streamlit run assign_2.py
```

Streamlit will print a local URL to open in your browser. In Codespaces, use the forwarded port link if prompted.

## 5) Troubleshooting

*  ModuleNotFoundError: No module named 'streamlit'
  Your `.venv` isn’t active or Streamlit isn’t installed. Activate the `.venv` and run `pip install streamlit`.
* App can’t find your keys
  Re-check you replaced every `put ____ key here` placeholder. If using env vars, confirm they’re set in the active shell (print them) and that the code reads from `os.environ`.
* Port issues
  Run on a different port:
  `streamlit run assign_2.py --server.port 8502`

# Configuration Changes Made
* **No other config changes** were required to the provided template. If your local environment differs, document additional edits here (e.g., pinned versions, port settings, or tool timeouts).
