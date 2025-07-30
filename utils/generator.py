import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(query, context, model="gpt-3.5-turbo"):
    system_prompt = "You are a helpful assistant. Answer questions using the provided context. If the answer isn't found, say so."

    user_prompt = f"""
Context:
{context}

Question: {query}
Answer:"""

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"
