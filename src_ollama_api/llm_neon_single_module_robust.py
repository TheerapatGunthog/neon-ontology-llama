#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeOn-style ontology induction from ONE CSV at a time.
Robust version: adaptive batching, strict schemas, multi-format fallbacks.

- Input: one CSV column (module) such as Skill or Subject.
- Output: Turtle (.ttl) for this module only.
- LLM: keep your chosen Ollama model. We harden prompts and parsing instead.

Usage:
  python llm_neon_single_module_robust.py \
    --input dataset/cleaned_Job_Skill.csv \
    --module Skill \
    --dataset ComputerIndustry \
    --out ./output/ontology_skill.ttl \
    --model gpt-oss:20b --temperature 0.1 --batch 120
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# ---------------------------- Defaults ----------------------------

MODEL_NAME = "llama3.1:latest"
OLLAMA_HOST = "http://localhost:11434"
TEMPERATURE = 0.2
NUM_CTX = 8192
MAX_TOKENS = 2048
DEFAULT_BATCH = 200  # requested batch size; may shrink adaptively
TRUNCATE_ITEM_CHARS = 160
MIN_BATCH = 12

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------- Ollama client ----------------------------


@dataclass
class OllamaConfig:
    model: str = MODEL_NAME
    host: str = OLLAMA_HOST
    keep_alive: str = "10m"
    temperature: float = TEMPERATURE
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.05
    num_ctx: int = NUM_CTX
    max_tokens: int = MAX_TOKENS


def _extract_top_level_json(s: str) -> Optional[str]:
    start = None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return s[start : i + 1]
    return None


def _strip_md(s: str) -> str:
    return re.sub(r"^```(?:json|txt)?|```$", "", s.strip(), flags=re.MULTILINE)


def _ollama_generate(
    cfg: OllamaConfig, prompt: str, options: Dict[str, Any], retries: int = 3
) -> str:
    url = f"{cfg.host}/api/generate"
    raw = ""
    for attempt in range(retries):
        opts = {
            "temperature": max(cfg.temperature * (0.6**attempt), 0.05),
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repeat_penalty": cfg.repeat_penalty,
            "num_ctx": cfg.num_ctx,
            "num_predict": cfg.max_tokens,
        }
        opts.update(options or {})
        payload = {
            "model": cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": opts,
            "keep_alive": cfg.keep_alive,
        }
        try:
            r = requests.post(url, json=payload, timeout=900)
            r.raise_for_status()
            data = r.json()
            raw = data.get("response", "")
            if raw:
                return raw
        except Exception as e:
            logging.warning(f"Ollama error attempt {attempt+1}: {e}")
            time.sleep(1 + attempt)
    return raw


# ---------------------------- IO utils ----------------------------


def to_upper_camel(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", " ", (s or ""))
    parts = [p for p in s.strip().split() if p]
    if not parts:
        return "Unnamed"
    out = "".join(w[:1].upper() + w[1:].lower() for w in parts)
    return re.sub(r"^[0-9]+", "", out) or "Unnamed"


def ttl_escape(s: str) -> str:
    return (s or "").replace('"', '\\"')


def _read_single_column_csv(path: Path, column: Optional[str]) -> List[str]:
    rows: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        field = column or (reader.fieldnames[0] if reader.fieldnames else None)
        if not field:
            return rows
        for r in reader:
            v = (r.get(field) or "").strip()
            if v:
                rows.append(v)
    return rows


# ---------------------------- NeOn prompts (improved) ----------------------------


def build_module_concept_prompt(
    dataset: str, module: str, role_hint: str, examples: List[str]
) -> str:
    system = (
        "You are an ontology engineer applying the NeOn Methodology for ONE module.\n"
        "Objective: derive a compact taxonomy for the given items and propose SKOS relations within THIS module only.\n"
        "Do not use other modules. Return STRICT JSON. No commentary."
    )
    user_spec = {
        "dataset": dataset,
        "module": module,
        "role_hint": role_hint,
        "neon_steps": [
            "Scope: limit to this module only.",
            "Reuse: only common-sense categories (no external URIs).",
            "Informal conceptualization: induce classes from examples.",
            "Intra-module relations: broader/narrower/related only.",
        ],
        "rules": {
            "class_count": "Choose K in [2, 20] based on variety.",
            "class_name": "UpperCamelCase ASCII. No punctuation.",
            "description": "≤ 20 words. Objective genus–differentia.",
            "relations": "Only link between class names that exist.",
            "json_schema": {
                "top_class": "str",
                "classes": [
                    {"cluster_id": "int", "class_name": "str", "description": "str"}
                ],
                "relations": [
                    {
                        "source": "str",
                        "target": "str",
                        "type": "broader|narrower|related",
                    }
                ],
            },
        },
        "examples_sample": examples[:30],
        "required_output": ["top_class", "classes", "relations"],
    }
    prompt = (
        "Respond with JSON only. No prose.\n\n"
        f"[SYSTEM]\n{system}\n[/SYSTEM]\n"
        "[OUTPUT_FORMAT]\n"
        '{"top_class": str, "classes": [{"cluster_id": int, "class_name": str, "description": str}], '
        '"relations": [{"source": str, "target": str, "type": "broader|narrower|related"}]}\n'
        "[/OUTPUT_FORMAT]\n"
        f"[USER]\nINPUT_JSON:\n{json.dumps(user_spec, ensure_ascii=False)}\n[/USER]"
    )
    return prompt


def build_population_prompts(
    dataset: str,
    module: str,
    top_class: str,
    classes: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
) -> Tuple[str, str]:
    class_min = [
        {"cluster_id": int(c["cluster_id"]), "class_name": c["class_name"]}
        for c in classes
    ]
    schema = {
        "dataset": dataset,
        "module": module,
        "top_class": top_class,
        "classes": class_min,
        "items": items,
        "rules": {
            "assignment": "Exactly one cluster_id per item.",
            "json_schema": {"assignments": [{"row": "int", "cluster_id": "int"}]},
        },
        "required_output": ["assignments"],
    }
    json_prompt = (
        "Respond with JSON only. No prose.\n"
        "[TASK] Assign each item to exactly one class by cluster_id for THIS module only.[/TASK]\n"
        "[OUTPUT_FORMAT]\n"
        '{"assignments": [{"row": int, "cluster_id": int}]}\n'
        "[/OUTPUT_FORMAT]\n"
        f"[USER]\nINPUT_JSON:\n{json.dumps(schema, ensure_ascii=False)}\n[/USER]"
    )
    tsv_prompt = (
        "Respond with only lines in the format 'row<TAB>cluster_id'. No JSON. No extra text.\n"
        "Example:\n0\t2\n1\t0\n2\t1\n"
        f"Classes: {json.dumps(class_min, ensure_ascii=False)}\n"
        f"Items: {json.dumps(items, ensure_ascii=False)}\n"
        "Output now."
    )
    return json_prompt, tsv_prompt


# ---------------------------- Parsing helpers ----------------------------


def parse_assignments_json(text: str) -> Optional[List[Dict[str, int]]]:
    if not text:
        return None
    text = _strip_md(text)
    obj_text = _extract_top_level_json(text) or text
    try:
        data = json.loads(obj_text)
    except Exception:
        return None
    arr = data.get("assignments")
    if not isinstance(arr, list):
        return None
    out = []
    for a in arr:
        if not isinstance(a, dict):
            return None
        if "row" not in a or "cluster_id" not in a:
            return None
        out.append({"row": int(a["row"]), "cluster_id": int(a["cluster_id"])})
    return out


def parse_assignments_tsv(text: str) -> Optional[List[Dict[str, int]]]:
    if not text:
        return None
    text = _strip_md(text)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(-?\d+)\s+(-?\d+)$", line)
        if not m:
            return None
        r = int(m.group(1))
        c = int(m.group(2))
        out.append({"row": r, "cluster_id": c})
    return out if out else None


def fill_missing(
    index_range: List[int], parsed: List[Dict[str, int]], default_cid: int = 0
) -> List[int]:
    pos_map = {r: i for i, r in enumerate(index_range)}
    out = [-1] * len(index_range)
    for a in parsed:
        r = a["row"]
        if r in pos_map:
            out[pos_map[r]] = a["cluster_id"]
    for i, v in enumerate(out):
        if v == -1:
            out[i] = default_cid
    return out


# ---------------------------- TTL rendering ----------------------------


def render_module_ttl(
    dataset: str,
    module: str,
    top_class: str,
    classes: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    items: List[str],
    assignments: List[int],
) -> str:
    M = to_upper_camel(module)
    ttl: List[str] = [
        "@prefix :    <http://example.org/onto#> .",
        "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix owl:  <http://www.w3.org/2002/07/owl#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "",
        ": a owl:Ontology ;",
        f'  rdfs:label "{dataset} {M} Ontology"@en ;',
        f'  rdfs:comment "NeOn single-module induction for {M}."@en .',
        "",
        f':{M} a owl:Class ; rdfs:label "{ttl_escape(M)}"@en .',
        "",
    ]

    for c in classes:
        name = to_upper_camel(str(c["class_name"]))
        desc = ttl_escape(c.get("description", ""))
        ttl += [
            f":{name} a owl:Class ;",
            f'  rdfs:label "{name}"@en ;',
            f'  rdfs:comment "{desc}"@en ;',
            f"  rdfs:subClassOf :{M} .",
            "",
        ]

    for rel in relations:
        s = to_upper_camel(str(rel.get("source", "")))
        t = to_upper_camel(str(rel.get("target", "")))
        typ = str(rel.get("type", "related")).lower()
        pred = {
            "broader": "skos:broader",
            "narrower": "skos:narrower",
            "related": "skos:related",
        }.get(typ, "skos:related")
        if s and t:
            ttl.append(f":{s} {pred} :{t} .")
    ttl.append("")

    for i, text in enumerate(items):
        label = to_upper_camel(f"{M}_{i+1}")
        ttl += [
            f":{label} a :{M} ;",
            f'  rdfs:label "{ttl_escape(text)}"@en .',
        ]
    ttl.append("")

    for i, cid in enumerate(assignments):
        label = to_upper_camel(f"{M}_{i+1}")
        cls = None
        for c in classes:
            if int(c["cluster_id"]) == int(cid):
                cls = to_upper_camel(str(c["class_name"]))
                break
        if cls:
            ttl.append(f":{label} a :{cls} .")
    ttl.append("")

    return "\n".join(ttl)


# ---------------------------- Main flow ----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--module", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--column", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--ollama", default=OLLAMA_HOST)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--test_rows", type=int, default=None)
    args = ap.parse_args()

    cfg = OllamaConfig(
        model=args.model,
        host=args.ollama,
        temperature=args.temperature,
        num_ctx=NUM_CTX,
        max_tokens=args.max_tokens,
    )

    items = _read_single_column_csv(args.input, args.column)
    if args.test_rows is not None:
        items = items[: args.test_rows]
    logging.info(f"Loaded {len(items)} rows from {args.input}")

    role_hint = "Skill list" if "skill" in args.module.lower() else "Subject name"
    concept_prompt = build_module_concept_prompt(
        args.dataset, args.module, role_hint, items
    )
    raw_concept = _ollama_generate(cfg, concept_prompt, {"format": "json"}, retries=4)
    concept = {}
    try:
        obj = _extract_top_level_json(_strip_md(raw_concept)) or raw_concept
        concept = json.loads(obj)
    except Exception:
        logging.error("Concept JSON parse failed. Using 2-class fallback.")
        concept = {}
    top_class = concept.get("top_class") or to_upper_camel(args.module)
    classes = concept.get("classes") or [
        {
            "cluster_id": 0,
            "class_name": f"{to_upper_camel(args.module)}General",
            "description": "Generic bucket",
        },
        {
            "cluster_id": 1,
            "class_name": f"{to_upper_camel(args.module)}Specific",
            "description": "Specific bucket",
        },
    ]
    relations = concept.get("relations") or []

    n = len(items)
    assignments: List[int] = [-1] * n
    batch_size = max(MIN_BATCH, min(args.batch, n if n > 0 else DEFAULT_BATCH))

    with tqdm(total=n, desc="Populate module", unit="row") as pbar:
        start = 0
        while start < n:
            end = min(start + batch_size, n)
            index_range = list(range(start, end))
            batch = [
                {
                    "row": j,
                    "text": (
                        items[j]
                        if len(items[j]) <= TRUNCATE_ITEM_CHARS
                        else items[j][:TRUNCATE_ITEM_CHARS]
                    ),
                }
                for j in index_range
            ]

            json_prompt, tsv_prompt = build_population_prompts(
                args.dataset, args.module, top_class, classes, batch
            )

            raw_json = _ollama_generate(cfg, json_prompt, {"format": "json"}, retries=4)
            parsed = parse_assignments_json(raw_json)

            if parsed is None:
                raw_tsv = _ollama_generate(
                    cfg, tsv_prompt, {"format": "text"}, retries=3
                )
                parsed = parse_assignments_tsv(raw_tsv)

            if parsed is None:
                if batch_size > MIN_BATCH:
                    logging.warning(
                        f"Invalid 'assignments' for rows [{start}:{end}]. Shrink batch and retry."
                    )
                    batch_size = max(MIN_BATCH, batch_size // 2)
                    continue
                else:
                    logging.error(
                        f"Invalid 'assignments' for rows [{start}:{end}] at MIN_BATCH. Assign all to 0."
                    )
                    for j in index_range:
                        assignments[j] = 0
                    pbar.update(len(index_range))
                    start = end
                    continue

            filled = fill_missing(index_range, parsed, default_cid=0)
            for pos, j in enumerate(index_range):
                assignments[j] = int(filled[pos])

            pbar.update(len(index_range))
            start = end

            if batch_size < args.batch:
                batch_size = min(args.batch, batch_size * 2)

    ttl = render_module_ttl(
        dataset=args.dataset,
        module=args.module,
        top_class=top_class,
        classes=classes,
        relations=relations,
        items=items,
        assignments=assignments,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(ttl, encoding="utf-8")
    print(str(args.out))


if __name__ == "__main__":
    main()
