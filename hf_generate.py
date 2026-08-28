from dotenv import load_dotenv
from openai import OpenAI
import os

# Load variables from .env
load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Correct the grammar: 'Thnks for contacting us.'"
        }
    ],
    temperature=0
)

print(response.choices[0].message.content)