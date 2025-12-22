from datasets import load_dataset
from dotenv import load_dotenv

from athena_dr import AnswerType, DeepResearchAgent
from athena_dr.utils import get_config

load_dotenv()
config = get_config("configs/deep_research.yml")
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
agent = DeepResearchAgent(config=config)
agent.generate_sft_traces(
    dataset=dataset,
    answer_type=AnswerType.EXACT,
    prompt_column="question",
    answer_column="answer",
    max_examples=5,
    dataset_name="AthenaAgent42/Hotpotqa-traces",
)
