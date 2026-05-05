from typing import List


def simple_llm_generate(query: str, contexts: List[str]) -> str:
    context_preview = "\n---\n".join(contexts[:3])
    return (
        f"Question: {query}\n\n"
        f"Answer (mocked):\n"
        f"Based on the following retrieved context:\n{context_preview}\n\n"
        f"(Replace this with real LLM API call later.)"
    )
