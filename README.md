# 🚀 Free LLM API Router

**Route your requests across multiple free LLM providers with automatic failover.**

Zero cost. No API key required for basic usage. Supports OpenAI-compatible format.

## ✨ Features

- 🔄 **Auto-failover** — if one provider is down/rate-limited, routes to next
- 🎯 **OpenAI-compatible** — drop-in replacement for any OpenAI SDK
- 📊 **Load balancing** — round-robin across providers
- 🛡️ **Rate limit detection** — auto-rotates on 429s
- 🔌 **Plugin architecture** — add your own providers easily
- 📈 **Usage tracking** — see which providers you're using

## 🏗️ Supported Free Providers

| Provider | Models | Rate Limit | Auth |
|----------|--------|------------|------|
| Mistral AI | mistral-small, codestral, etc | 1B tokens/month | API Key (free signup) |
| Google AI Studio | Gemini 2.5 Flash, Pro | 15 RPM / 1M TPD | API Key (free) |
| Groq | Llama 3.3, Mixtral, Gemma | 30 RPM / 14K TPM | API Key (free) |
| Cerebras | Llama 3.3 70B | 30 RPM | API Key (free) |
| SambaNova | Llama 3.3, DeepSeek | 100K tokens/day | API Key (free) |
| Together AI | Llama, Mistral, Qwen | $1 free credits | API Key (free) |
| Fireworks AI | Llama, Mixtral, DeepSeek | $1 free credits | API Key (free) |
| Chutes.ai | Various open models | Free tier | API Key (free) |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/setan21/free-llm-api-router.git
cd free-llm-api-router

# Install
pip install -r requirements.txt

# Configure (add your free API keys)
cp config.example.yaml config.yaml
# Edit config.yaml with your keys

# Run
python server.py --port 8080

# Use it like OpenAI
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## 📖 Configuration

```yaml
# config.yaml
providers:
  - name: mistral
    base_url: https://api.mistral.ai/v1
    api_key: YOUR_FREE_KEY
    models: [mistral-small-latest, codestral-latest]
    priority: 1

  - name: groq
    base_url: https://api.groq.com/openai/v1
    api_key: YOUR_FREE_KEY
    models: [llama-3.3-70b-versatile, mixtral-8x7b-32768]
    priority: 2

  - name: google
    base_url: https://generativelanguage.googleapis.com/v1beta/openai
    api_key: YOUR_FREE_KEY
    models: [gemini-2.5-flash, gemini-2.5-pro]
    priority: 3

routing:
  strategy: round-robin  # or: priority, random, least-latency
  failover: true
  max_retries: 3
  timeout: 30
```

## 🐍 Python SDK Usage

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="auto",  # Routes to best available provider
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
```

## 🔧 Add Custom Provider

```python
# providers/my_provider.py
from .base import BaseProvider

class MyProvider(BaseProvider):
    name = "my-provider"
    base_url = "https://api.example.com/v1"
    
    async def is_available(self) -> bool:
        # Check rate limits, balance, etc.
        return True
```

## 📊 Dashboard

Access the built-in dashboard at `http://localhost:8080/dashboard`:

- Provider status (up/down/rate-limited)
- Request counts per provider
- Latency stats
- Failover events

## 🤝 Contributing

1. Fork it
2. Create your branch (`git checkout -b feature/new-provider`)
3. Commit (`git commit -m 'Add new provider'`)
4. Push (`git push origin feature/new-provider`)
5. Open a PR

## ⭐ Star History

If this project saved you money on API costs, give it a ⭐!

## 📄 License

MIT
