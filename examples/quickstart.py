"""A multi-turn conversation with a TRACE model.

Directly against vLLM (this package does the formatting and parsing):
    python examples/quickstart.py --base-url http://localhost:30003/v1 --model Dipto084/Llama3.1-8B-TRACE

Through the proxy (plain OpenAI client, no TRACE-specific code):
    python examples/quickstart.py --base-url http://localhost:8000/v1 --model Dipto084/Llama3.1-8B-TRACE --proxy
"""

import argparse

TURNS = [
    "I'm planning a two-week trip to Japan in November. Where should I start?",
    "Let's say Tokyo and Kyoto. How many days each?",
    "What's the etiquette I should know before visiting temples?",
]


def via_package(base_url: str, model: str) -> None:
    from trace_defense import Conversation, TraceModel

    conv = Conversation(TraceModel(model=model, base_url=base_url))
    for user_msg in TURNS:
        resp = conv.send(user_msg)
        print(f"USER: {user_msg}")
        print(f"[STATE] action={resp.action} jailbreak_score={resp.jailbreak_score}")
        print(f"ASSISTANT: {resp.answer}\n")


def via_proxy(base_url: str, model: str) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    history = []
    for user_msg in TURNS:
        history.append({"role": "user", "content": user_msg})
        completion = client.chat.completions.create(model=model, messages=history)
        answer = completion.choices[0].message.content
        history.append({"role": "assistant", "content": answer})
        print(f"USER: {user_msg}")
        print(f"ASSISTANT: {answer}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:30003/v1")
    p.add_argument("--model", default="Dipto084/Llama3.1-8B-TRACE")
    p.add_argument("--proxy", action="store_true", help="talk to the proxy with a plain OpenAI client")
    a = p.parse_args()
    (via_proxy if a.proxy else via_package)(a.base_url, a.model)
