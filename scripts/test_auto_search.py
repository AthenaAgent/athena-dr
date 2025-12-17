import asyncio
import os
import warnings

# Set before importing tokenizers to avoid deadlock warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import weave
from dotenv import load_dotenv

from athena_dr.agent.shared_prompts import (
    SHORT_FORM_ANSWER_EVALUATION_USER_PROMPT_FORMAT,
)
from athena_dr.agent.workflows import AutoReasonSearchWorkflow

warnings.filterwarnings("ignore")

# Get the project root directory (parent of scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env from project root explicitly
env_path = os.path.join(PROJECT_ROOT, ".env")
print(f"Loading .env from: {env_path}")
print(f".env exists: {os.path.exists(env_path)}")
load_dotenv(env_path)

# Debug: verify API key is loaded
api_key = os.environ.get("OPENROUTER_API_KEY", "")
if api_key:
    print(f"✓ OPENROUTER_API_KEY loaded: {api_key[:15]}...{api_key[-4:]} (length: {len(api_key)})")
else:
    print("✗ OPENROUTER_API_KEY is NOT SET!")
    # Check if it's in shell env but not .env
    import subprocess
    result = subprocess.run(['printenv', 'OPENROUTER_API_KEY'], capture_output=True, text=True)
    if result.stdout.strip():
        print(f"  But found in shell: {result.stdout.strip()[:15]}...{result.stdout.strip()[-4:]}")

# Enable litellm debug mode after loading env vars
import litellm
litellm._turn_on_debug()


@weave.op
def run_workflow(problem: str):
    workflow = AutoReasonSearchWorkflow(
        configuration=os.path.join(PROJECT_ROOT, "configs/auto_search_configs/openrouter.yml")
        # configuration="configs/auto_search_configs/azure-gpt4.yml"
    )
    result = asyncio.run(workflow(problem=problem, dataset_name="short_form"))
    return result


if __name__ == "__main__":
    weave.init(project_name="athena_dr")
    # problem = "According to the Wikipedia article on Academic publishing, what dispute rate did Robert K. Merton report for simultaneous discoveries in the 17th century?"
    problem = "The Oberoi family is part of a hotel company that has a head office in what city?"
    run_workflow(
        problem=SHORT_FORM_ANSWER_EVALUATION_USER_PROMPT_FORMAT.format(prompt=problem)
    )
