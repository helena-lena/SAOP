import requests
import ast
import json

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
    try:
        output = output.replace("'", '"')
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
    return None

'''def extract_all_json_blocks(text: str, marker="json") -> list:
    """
    Extract all JSON blocks from text considering nested braces.
    Captures from first '{' after marker until braces are balanced.
    """
    json_blocks = []
    idx = 0
    while True:
        start_marker = text.find(marker, idx)
        if start_marker == -1:
            break
        brace_start = text.find('{', start_marker)
        if brace_start == -1:
            break

        stack = []
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                stack.append('{')
            elif text[i] == '}':
                stack.pop()
                if not stack:
                    json_str = text[brace_start:i+1]
                    json_blocks.append(json_str.strip())
                    idx = i + 1
                    break
        else:
            # Unclosed braces
            break
    return json_blocks'''

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