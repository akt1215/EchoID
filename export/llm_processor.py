import requests
import json

class LLMProcessor:
    def __init__(self, model_name="minimax-m3:cloud", timeout=120):
        self.model_name = model_name
        self.timeout = timeout
        self.api_url = "http://localhost:11434/api/generate"

    def generate_summary(self, transcript_text):
        """
        Sends the transcript to Ollama and requests a structured JSON summary.
        """
        prompt = f"""
        You are an AI meeting assistant. Read the meeting transcript and return a structured summary as strictly valid JSON. Output only the raw JSON object, with no markdown fences.

        The transcript is annotated with **Speaker** (MM:SS): markers. For every summary point and action item, cite the (MM:SS) timestamp of the marker nearest to where it was discussed. Use "" when no timestamp applies.

        The JSON object must have exactly these three keys:
        1. "executive_summary": 3 to 6 objects, each {{"point": "<a main topic>", "timestamp": "MM:SS"}}.
        2. "action_items": a list of objects {{"item": "<a specific task and its assignee>", "timestamp": "MM:SS"}}.
        3. "entities": a list of strings — names of people, projects, and clients mentioned.

        Transcript:
        {transcript_text}
        """

        # NOTE: deliberately not setting Ollama's "format": "json". That
        # constrains the token grammar and breaks reasoning models (e.g.
        # gpt-oss), which leak their chain-of-thought into the response and
        # never emit clean JSON. With a plain prompt these models route
        # reasoning to a separate "thinking" field and return valid JSON.
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        print(f"Sending transcript to Ollama ({self.model_name}) for summarization...")
        try:
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            result_json = response.json()
            response_text = result_json.get("response", "")

            data = self._parse_json_object(response_text)
            if data is not None:
                return {
                    "executive_summary": [self._normalize_entry(e, "point")
                                          for e in data.get("executive_summary", [])],
                    "action_items": [self._normalize_entry(e, "item")
                                     for e in data.get("action_items", [])],
                    "entities": [str(x).strip() for x in data.get("entities", []) if str(x).strip()],
                }
            print("Error: Could not parse JSON from LLM response.")
            print(response_text)
            return {
                "executive_summary": [{"text": "Error parsing summary.", "timestamp": None}],
                "action_items": [],
                "entities": []
            }

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return {
                "executive_summary": [{"text": f"Error connecting to LLM: {e}", "timestamp": None}],
                "action_items": [],
                "entities": []
            }

    @staticmethod
    def _clean_ts(ts):
        """Return ts if it looks like MM:SS (or M:SS), else None."""
        if not ts:
            return None
        s = str(ts).strip()
        parts = s.split(":")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            return s
        return None

    @staticmethod
    def _normalize_entry(entry, key):
        """Coerce a summary/action entry to {"text", "timestamp"}, tolerating
        either an object {<key>/"text", "timestamp"} or a bare string."""
        if isinstance(entry, dict):
            text = entry.get(key) or entry.get("text") or ""
            return {"text": str(text).strip(), "timestamp": LLMProcessor._clean_ts(entry.get("timestamp"))}
        return {"text": str(entry).strip(), "timestamp": None}

    @staticmethod
    def _parse_json_object(text):
        """Parse a JSON object from an LLM response.

        Tries a direct parse first, then falls back to extracting the first
        balanced {...} object. Unlike a naive regex, the brace scan handles
        nested objects, so it survives a model that prepends prose or wraps
        the JSON in a ```json fence. Returns None if nothing parses.
        """
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None
