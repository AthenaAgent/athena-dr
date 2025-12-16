import asyncio
import warnings

import weave
from dotenv import load_dotenv

from athena_dr.agent.shared_prompts import (
    SHORT_FORM_ANSWER_EVALUATION_USER_PROMPT_FORMAT,
)
from athena_dr.agent.workflows import AutoReasonSearchWorkflow

warnings.filterwarnings("ignore")
load_dotenv()


@weave.op
def run_workflow(problem: str):
    workflow = AutoReasonSearchWorkflow(configuration="configs/auto_search_configs.yml")
    result = asyncio.run(workflow(problem=problem))
    return result


if __name__ == "__main__":
    weave.init(project_name="athena_dr")
    # problem = "According to the Wikipedia article on Academic publishing, what dispute rate did Robert K. Merton report for simultaneous discoveries in the 17th century?"
    problem = "Which magazine was started first Arthur's Magazine or First for Women?"
    run_workflow(
        problem=SHORT_FORM_ANSWER_EVALUATION_USER_PROMPT_FORMAT.format(prompt=problem)
    )
