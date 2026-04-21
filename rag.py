import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq


DEFAULT_MODEL = "llama-3.3-70b-versatile"


class RagError(Exception):
    """Raised when the vectorless RAG pipeline cannot complete."""


def load_tree(path: str | Path) -> dict[str, Any]:
    tree_path = Path(path)
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")
    return json.loads(tree_path.read_text(encoding="utf-8"))


def get_groq_client(api_key: str | None = None) -> Groq:
    load_dotenv()
    key = api_key or os.getenv("GRO_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        raise RagError(
            "Groq API key is missing. Add GRO_API_KEY or GROQ_API_KEY "
            "to .env or Streamlit secrets."
        )
    return Groq(api_key=key)


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}


def groq_json(
    client: Groq,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 350,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=max_tokens,
    )
    content = response.choices[0].message.content.strip()
    return parse_json_response(content)


def select_lessons(
    query: str,
    tree: dict[str, Any],
    client: Groq,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    lesson_index = []

    for lesson in tree["lessons"]:
        lesson_index.append(
            {
                "lesson_id": lesson["lesson_id"],
                "lesson_name": lesson["lesson_name"],
                "items_count": lesson["items_count"],
                "sample_questions": [
                    item["question"] for item in lesson["items"][:8]
                ],
            }
        )

    prompt = f"""
You are selecting the most relevant lessons from a Siddha medicine knowledge tree.

User query:
{query}

Lesson index:
{compact_json(lesson_index)}

Return ONLY valid JSON.

Format:
{{
  "thinking": "short reason",
  "lesson_ids": [1]
}}

Rules:
- Select 1 to 2 lesson_ids.
- Prefer the single most specific lesson when the query clearly belongs to one lesson.
- Choose lessons most likely to contain the answer.
- Do not create new lesson IDs.
- Keep thinking short.
"""

    result = groq_json(client, prompt, model=model, max_tokens=300)
    valid_ids = {lesson["lesson_id"] for lesson in tree["lessons"]}
    lesson_ids = [x for x in result.get("lesson_ids", []) if x in valid_ids]

    return {
        "thinking": result.get("thinking", ""),
        "lesson_ids": lesson_ids,
    }


def select_items(
    query: str,
    tree: dict[str, Any],
    lesson_ids: list[int],
    client: Groq,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    candidates = []

    for lesson in tree["lessons"]:
        if lesson["lesson_id"] in lesson_ids:
            for item in lesson["items"]:
                candidates.append(
                    {
                        "node_id": f"Q{item['id']}",
                        "id": item["id"],
                        "lesson_name": lesson["lesson_name"],
                        "question": item["question"],
                    }
                )

    if not candidates:
        return {"thinking": "No candidate lessons found.", "node_list": []}

    prompt = f"""
You are selecting relevant Q&A nodes from a Siddha medicine knowledge tree.

User query:
{query}

Candidate Q&A nodes:
{compact_json(candidates)}

Return ONLY valid JSON.

Format:
{{
  "thinking": "short reason",
  "node_list": ["Q10", "Q25"]
}}

Rules:
- Select 1 to 5 node_ids.
- Prefer exact question matches over broad related matches.
- Choose questions that directly match the user query.
- Do not create node_ids.
- If no relevant node exists, return an empty list.
- Keep thinking short.
"""

    result = groq_json(client, prompt, model=model, max_tokens=400)
    valid_node_ids = {item["node_id"] for item in candidates}
    node_list = [
        node_id for node_id in result.get("node_list", []) if node_id in valid_node_ids
    ]

    return {
        "thinking": result.get("thinking", ""),
        "node_list": node_list,
    }


def find_nodes_by_ids(
    tree: dict[str, Any],
    target_node_ids: list[str],
) -> list[dict[str, Any]]:
    found = []
    target_node_ids = set(target_node_ids)

    for lesson in tree["lessons"]:
        for item in lesson["items"]:
            node_id = f"Q{item['id']}"
            if node_id in target_node_ids:
                found.append(
                    {
                        "node_id": node_id,
                        "lesson_id": lesson["lesson_id"],
                        "lesson_name": lesson["lesson_name"],
                        "id": item["id"],
                        "title": item["question"],
                        "text": item["answer"],
                    }
                )

    return found


def generate_answer(
    query: str,
    nodes: list[dict[str, Any]],
    client: Groq,
    model: str = DEFAULT_MODEL,
) -> str:
    if not nodes:
        return "இந்த கேள்விக்கு பொருத்தமான தகவல் இந்த knowledge tree-ல் கிடைக்கவில்லை."

    context_parts = []
    for node in nodes:
        context_parts.append(
            f"[Node: {node['node_id']} | Lesson: {node['lesson_name']}]\n"
            f"Question: {node['title']}\n"
            f"Answer: {node['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a Siddha medicine assistant.

Answer the user using ONLY the provided context.

User query:
{query}

Context:
{context}

Rules:
- Answer in Tamil.
- Use only the provided context.
- Do not invent remedies.
- Mention the matching source node IDs.
- Add a short safety note recommending a qualified doctor or practitioner for serious symptoms.

Answer:
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=1000,
    )
    return response.choices[0].message.content.strip()


def vectorless_rag(
    query: str,
    tree: dict[str, Any],
    client: Groq,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    lesson_search = select_lessons(query, tree, client, model=model)
    item_search = select_items(
        query,
        tree,
        lesson_search["lesson_ids"],
        client,
        model=model,
    )
    nodes = find_nodes_by_ids(tree, item_search["node_list"])
    answer = generate_answer(query, nodes, client, model=model)

    return {
        "query": query,
        "lesson_reasoning": lesson_search["thinking"],
        "lesson_ids": lesson_search["lesson_ids"],
        "item_reasoning": item_search["thinking"],
        "node_ids": item_search["node_list"],
        "nodes": nodes,
        "answer": answer,
    }
