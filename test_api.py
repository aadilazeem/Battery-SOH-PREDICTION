import requests

api_key = "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"

response = requests.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": api_key,
        "Content-Type": "application/json"
    },
    json={
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}]
    }
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")