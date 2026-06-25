#!/usr/bin/env python3
"""
Auto-collect free API keys from LLM providers.
Generates a config.yaml with your keys.
"""

import json
import sys
import yaml

PROVIDERS = {
    "mistral": {
        "name": "Mistral AI",
        "signup": "https://console.mistral.ai",
        "models": ["mistral-small-latest", "codestral-latest", "open-mistral-nemo"],
        "base_url": "https://api.mistral.ai/v1",
        "limit": "1B tokens/month free",
    },
    "groq": {
        "name": "Groq",
        "signup": "https://console.groq.com",
        "models": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "base_url": "https://api.groq.com/openai/v1",
        "limit": "30 RPM, 14K TPM free",
    },
    "cerebras": {
        "name": "Cerebras",
        "signup": "https://cloud.cerebras.ai",
        "models": ["llama-3.3-70b", "llama-4-scout-17b-16e-instruct"],
        "base_url": "https://api.cerebras.ai/v1",
        "limit": "30 RPM free",
    },
    "sambanova": {
        "name": "SambaNova",
        "signup": "https://cloud.sambanova.ai",
        "models": ["Meta-Llama-3.3-70B-Instruct", "DeepSeek-V3-0324"],
        "base_url": "https://api.sambanova.ai/v1",
        "limit": "100K tokens/day free",
    },
    "together": {
        "name": "Together AI",
        "signup": "https://api.together.xyz",
        "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "Qwen/Qwen3-235B-A22B-fp8-turbo"],
        "base_url": "https://api.together.xyz/v1",
        "limit": "$1 free credits",
    },
    "fireworks": {
        "name": "Fireworks AI",
        "signup": "https://fireworks.ai",
        "models": ["accounts/fireworks/models/llama-v3p3-70b-instruct"],
        "base_url": "https://api.fireworks.ai/inference/v1",
        "limit": "$1 free credits",
    },
}


def main():
    print("🔑 Free LLM API Key Setup\n")
    print("Get your free API keys from these providers:\n")

    for key, info in PROVIDERS.items():
        print(f"  📌 {info['name']}")
        print(f"     Signup: {info['signup']}")
        print(f"     Free tier: {info['limit']}")
        print(f"     Models: {', '.join(info['models'][:2])}...")
        print()

    print("=" * 50)
    print("Paste your API keys (press Enter to skip):\n")

    providers = []
    for key, info in PROVIDERS.items():
        api_key = input(f"  {info['name']} API key (or Enter to skip): ").strip()
        if api_key:
            providers.append({
                "name": key,
                "base_url": info["base_url"],
                "api_key": api_key,
                "models": info["models"],
                "priority": len(providers) + 1,
            })

    if not providers:
        print("\n❌ No keys provided. Add them manually to config.yaml")
        return

    config = {
        "providers": providers,
        "routing": {
            "strategy": "round-robin",
            "failover": True,
            "max_retries": 3,
            "timeout": 30,
        },
    }

    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n✅ config.yaml created with {len(providers)} providers!")
    print("Run: python server.py")


if __name__ == "__main__":
    main()
