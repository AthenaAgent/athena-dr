import rich
from openai import OpenAI

client = OpenAI(
    base_url="https://trojanvectors--gptoss-inference-serve.modal.run/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50,
    timeout=600.0,
)

rich.print(response.choices[0].message.content)
rich.print(response.usage)
