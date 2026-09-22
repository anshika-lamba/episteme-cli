import json
import re
import os
import google.generativeai as genai
from groq import Groq

def extract_last_json(text):
    try:
        matches = re.findall(r'\{.*\}', text.replace('\n', ' '), re.DOTALL)
        return json.loads(matches[-1]) if matches else None
    except:
        return None

class GeminiProvider:
    def __init__(self, api_key=None, model_name="gemini-1.5-flash"):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt):
        try:
            res = self.model.generate_content(prompt)
            parsed = extract_last_json(res.text)
            return parsed if parsed else {"error": "parse_failure"}
        except Exception as e:
            return {"error": str(e)}

class GroqProvider:
    def __init__(self, api_key=None, model_name="llama3-8b-8192"):
        self.model_name = model_name
        self.init_error = None
        try:
            key = api_key or os.environ.get("GROQ_API_KEY")
            self.client = Groq(api_key=key)
        except Exception as e:
            self.init_error = str(e)

    def generate(self, prompt):
        if self.init_error:
            return {"error": f"Groq setup failed: {self.init_error}"}
        try:
            res = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.0
            )
            parsed = extract_last_json(res.choices[0].message.content)
            return parsed if parsed else {"error": "parse_failure"}
        except Exception as e:
            return {"error": str(e)}
