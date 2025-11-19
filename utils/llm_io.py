import requests
import ast
import json
import re

def ollama_generate(prompt, model="gemma3:27b-it-qat", image_list = None):
    """Generate response from Ollama API"""
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    if image_list:
        data["images"] = image_list
        
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        print("Success")
    else:
        print(f"Error: {response.status_code}")
    return response.text

def check_json_format(output: str) -> dict | None:
    """Check if string is valid JSON format and convert to dict"""
    # ---------------------------------------
    # 1) Try strict JSON parsing first
    # ---------------------------------------
    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # ---------------------------------------
    # 2) Try Python literal parsing (single-quoted dict)
    # ---------------------------------------
    try:
        parsed = ast.literal_eval(output)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # ---------------------------------------
    # 3) Manual repair:
    # Convert python-style dict → JSON-style dict
    # Without touching apostrophes inside strings
    # ex) 'Show patient's diagnosis' --> "Show patient's diagnosis"
    # ---------------------------------------

    text = output

    # 3-1) Replace keys: 'key':  →  "key":
    text = re.sub(r"'([A-Za-z0-9_ ]+)'\s*:", r'"\1":', text)

    # 3-2) Replace values: 'value' → "value"
    #     But keep apostrophes inside value strings
    def replace_value_quotes(match):
        inner = match.group(1)
        # Escape existing double quotes inside the string
        inner = inner.replace('"', '\\"')
        return f"\"{inner}\""

    text = re.sub(r":\s*'([^']*)'", lambda m: replace_value_quotes(m), text)

    # ---------------------------------------
    # 4) Try parsing the repaired string
    # ---------------------------------------
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        print("[ParseError after repair]", e)
        print("[Debug repaired JSON text]:")
        print(text)

    return None

    '''try:
        #output = output.replace("'", '"')
        parsed_result = json.loads(output)
        if isinstance(parsed_result, dict):
            return parsed_result
    except json.JSONDecodeError:
        try:
            # Parse as Python literal (allows single quotes)
            parsed_result = ast.literal_eval(output)
            if isinstance(parsed_result, dict):
                return parsed_result
        except (ValueError, SyntaxError) as e:
            print(f"[ParseError] {e}")
    return None'''

def extract_all_json_blocks(text: str, marker="json") -> list:
    """
    Extract all JSON blocks from text.
    Handles markdown code blocks (closed or unclosed) and inline JSON.
    """
    import re
    json_blocks = []
    
    # Remove markdown code block markers first
    # This handles both ```json...``` and ```json... (unclosed)
    cleaned_text = re.sub(r'```json\s*\n?', '', text)
    cleaned_text = re.sub(r'\n?```', '', cleaned_text)
    
    # Now find all JSON objects (properly nested braces)
    idx = 0
    while True:
        brace_start = cleaned_text.find('{', idx)
        if brace_start == -1:
            break

        stack = []
        for i in range(brace_start, len(cleaned_text)):
            if cleaned_text[i] == '{':
                stack.append('{')
            elif cleaned_text[i] == '}':
                stack.pop()
                if not stack:
                    json_str = cleaned_text[brace_start:i+1]
                    json_blocks.append(json_str.strip())
                    idx = i + 1
                    break
        else:
            # Unclosed braces - try to find the end
            break
    
    return json_blocks