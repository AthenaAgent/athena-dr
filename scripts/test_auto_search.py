import weave
from dotenv import load_dotenv

from athena_dr.agent.deep_research import DeepResearchAgent
from athena_dr.utils import get_config

load_dotenv()
weave.init(project_name="athena-dr")
config = get_config("configs/deep_research.yml")
agent = DeepResearchAgent(config=config)
agent.predict("Which team does Lionel Messi play for as of December 2025?")
