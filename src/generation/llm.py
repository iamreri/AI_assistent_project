import os
import requests
import uuid
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_gigachat_token():
    auth_key = os.getenv("GIGACHAT_API_KEY")
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Bearer {auth_key}"
    }
    response = requests.post(url, data="scope=GIGACHAT_API_PERS", headers=headers, verify=False)
    response.raise_for_status()
    return response.json()["access_token"]

class LLMGenerator:
    def __init__(self):
        token = get_gigachat_token()
        self.client = OpenAI(
            api_key=token,
            base_url="https://api.giga.chat/v1",
            http_client=httpx.Client(verify=False)
        )
        self.model = "GigaChat-2"

    def generate(self, query: str, chunks: list):
        context_text = "\n\n".join([f"[{c['metadata'].get('document_title', 'Документ')}]: {c['text']}" for c in chunks])
        system_prompt = "Ты — ассистент для студентов. Отвечай на вопрос, используя ТОЛЬКО предоставленный контекст. Если ответа нет, скажи 'Я не знаю'."
        user_prompt = f"Контекст:\n{context_text}\n\nВопрос: {query}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
