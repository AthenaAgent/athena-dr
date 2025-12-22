import weave
from dotenv import load_dotenv

from athena_dr import AnswerType, DeepResearchAgent
from athena_dr.utils import get_config

load_dotenv()
weave.init(project_name="athena-dr")
config = get_config("configs/deep_research.yml")
agent = DeepResearchAgent(config=config)
agent.predict(
    query="According to the Wikipedia article on Academic publishing, what dispute rate did Robert K. Merton report for simultaneous discoveries in the 17th century?",
    # query="Calculate tariffs that should be paid for each small package under 800 us dollars from China to USA after 2025-6-1. Give tariff values for packages declared as $5, $10, $20, $50, $100, $500",
    # query="Has microwave-assisted fermentation of feather hydrolysate with brewer's spent grain been explored for co-production of peptides and bioenergy? Find prior work and reserach gap",
    # query="what are all the teams that Lionel Messi has played for?",
    answer_type=AnswerType.EXACT,
)
