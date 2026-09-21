import json
import re
import os
import google.generativeai as genai

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
