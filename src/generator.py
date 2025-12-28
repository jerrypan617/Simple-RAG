import os
from openai import OpenAI

class Generator:
    def __init__(self, api_key=None, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen2.5-7b-instruct-1m"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, query, context, system_prompt="You are a helpful assistant.", stream=True):
        full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nAccording to the contents above, answer the question in brief."
        messages = [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": query},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream
            )
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
