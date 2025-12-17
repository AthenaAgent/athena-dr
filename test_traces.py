import os

import rich
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)

response = client.chat.completions.create(
    model="z-ai/glm-4.6",
    messages=[
        {"role": "user", "content": "Compare photosynthesis vs cellular respiration."}
    ],
    extra_body={
        "reasoning": {"enabled": True},
        "provider": {
            "sort": "throughput",
        },
    },
)

rich.print(response.choices[0].message.reasoning_details[0]["text"])
