#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeOn-style multi-column ontology induction from separate CSV files.
- No argparse for dataset paths. Configure paths in FILES below.
- Robust JSON parsing with retries, JSON mode, sanitizer, and backoff.
"""

from __future__ import annotations

import json
import re
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------- Configuration ----------------------------

# Set your file paths here. Each CSV should contain a single relevant column.
# If there are multiple columns, specify the exact column name in FILE_COLUMNS.
FILES = {
    "subject_name_en": "./dataset/cleaned_Subject.csv",
    "Job_Title": "./dataset/cleaned_Job_Title.csv",
    "Job_Type": "./dataset/cleaned_Job_Type.csv",
    "Skill": "./dataset/cleaned_Job_Skill.csv",
}

# If the CSV has multiple columns, map which column to read. If missing, uses the first column.
FILE_COLUMNS: Dict[str, Optional[str]] = {
    "subject_name_en": None,
    "Job_Title": None,
    "Job_Type": None,
    "Skill": None,
}

# Ontology output path
OUT_TTL = "./output/neon_multicol_ontology.ttl"

# LLM runtime configuration
MODEL_NAME = "gpt-oss:20b"  # change to a stable JSON-friendly model if needed
OLLAMA_HOST = "http://localhost:11434"
TEMPERATURE = 0.2
NUM_CTX = 8192
MAX_TOKENS = 2048
MAX_RETRIES = 4
BATCH_SIZE = 100  # population batch per column
TRUNCATE_ITEM_CHARS = 120  # truncate each item text to avoid context overflow

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
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            if start is None:
                start = i
            depth += 1
        elif ch in "}]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return s[start : i + 1]
    return None


def _sanitize_json_str(s: str) -> str:
    s = s.strip().lstrip("\ufeff")
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].strip().lower().startswith("```json"):
            lines = lines[1:]
        elif lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    candidate = _extract_top_level_json(s)
    if candidate:
        s = candidate
    s = re.sub(r"\bNaN\b", "null", s)
    s = re.sub(r"\bInfinity\b", "null", s)
    s = re.sub(r"\b-?Inf\b", "null", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _ollama_chat_json(
    cfg: OllamaConfig,
    system: str,
    user: str,
    max_tokens: int = MAX_TOKENS,
    retries: int = MAX_RETRIES,
):
    url = f"{cfg.host}/api/generate"
    base_opts = {
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "repeat_penalty": cfg.repeat_penalty,
        "num_ctx": cfg.num_ctx,
        "num_predict": max_tokens,
    }
    for attempt in range(retries):
        sys_prompt = (
            system
            + "\nFormat: Output a single minified JSON object. No commentary or markdown."
        )
        if attempt > 0:
            sys_prompt += "\nSTRICT: Output ONLY valid JSON. If unsure, return {}."
        prompt = (
            "Respond with STRICT JSON only. No commentary.\n\n"
            f"[SYSTEM]\n{sys_prompt}\n[/SYSTEM]\n"
            f"[USER]\n{user}\n[/USER]"
        )
        opts = dict(base_opts)
        opts["temperature"] = max(cfg.temperature * (0.5**attempt), 0.05)
        if attempt >= 1:
            opts["format"] = "json"  # enable Ollama JSON mode from 2nd try

        payload = {
            "model": cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": opts,
            "keep_alive": cfg.keep_alive,
        }

        raw = ""
        try:
            r = requests.post(url, json=payload, timeout=1200)
            if r.status_code == 404:
                raise SystemExit(f"404 {url}. Check model '{cfg.model}' on {cfg.host}.")
            r.raise_for_status()
            raw = (r.json().get("response") or "").strip()
        except Exception as e:
            logging.warning(f"Ollama HTTP error (attempt {attempt+1}/{retries}): {e}")

        if not raw:
            wait = 1.5**attempt
            logging.warning(
                f"Ollama returned empty response (attempt {attempt+1}/{retries}). Sleeping {wait:.1f}s."
            )
            time.sleep(wait)
            continue

        text = _sanitize_json_str(raw)
        try:
            return json.loads(text)
        except Exception as e:
            logging.warning(
                f"Ollama JSON parse failed (attempt {attempt+1}/{retries}): {e}"
            )
            dbg = Path(f"./output/_ollama_badjson_attempt{attempt+1}_len{len(raw)}.txt")
            dbg.parent.mkdir(parents=True, exist_ok=True)
            try:
                dbg.write_text(raw, encoding="utf-8")
            except Exception:
                pass
            time.sleep(1.5**attempt)
            continue

    logging.error("Giving up parsing JSON. Returning empty dict {}.")
    return {}


# ---------------------------- Helpers ----------------------------

_CAMEL_RE = re.compile(r"[^A-Za-z0-9]+")


def to_upper_camel(s: str) -> str:
    parts = [p for p in _CAMEL_RE.split(s) if p]
    if not parts:
        return "Concept"
    return "".join(p.capitalize() for p in parts)[:80]


def ttl_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------- NeOn prompts ----------------------------


def build_column_concept_prompt(
    dataset: str, column_name: str, role_hint: str, examples: List[str]
) -> Tuple[str, str]:
    system = (
        "You are an ontology engineer applying the NeOn Methodology per column. "
        "Goal: conceptualize a small taxonomy for ONE column and propose SKOS relations within that column. "
        "Return STRICT JSON."
    )
    user_spec = {
        "dataset": dataset,
        "column": column_name,
        "role_hint": role_hint,
        "neon_steps": [
            "Scope & reuse check (implicit).",
            "Informal conceptualization: induce concepts from this column only.",
            "Identify intra-column taxonomic and associative relations.",
        ],
        "rules": {
            "n_classes": "Decide K in [2, 20] by variety.",
            "class_name": "UpperCamelCase ASCII. Short and specific.",
            "description": "≤ 20 words, objective definition.",
            "relations": "Use cluster_id indices. type ∈ {broader|narrower|related}.",
            "json_schema": {
                "classes": [
                    {"cluster_id": "int", "class_name": "str", "description": "str"}
                ],
                "relations": [
                    {
                        "source": "int",
                        "target": "int",
                        "type": "broader|narrower|related",
                    }
                ],
            },
        },
        "examples_sample": examples[:60],
        "required_output": ["classes", "relations"],
    }
    user = (
        "Induce per-column concept schema and intra-column relations. "
        "Return JSON with 'classes' and 'relations'.\n\n"
        f"INPUT_JSON:\n{json.dumps(user_spec, ensure_ascii=False)}"
    )
    return system, user


def build_cross_column_mapping_prompt(
    dataset: str, column_summaries: List[Dict[str, Any]]
) -> Tuple[str, str]:
    system = (
        "You are an ontology engineer applying the NeOn Methodology across columns. "
        "Propose cross-column object properties and simple constraints. "
        "Return STRICT JSON."
    )
    schema = []
    for cs in column_summaries:
        schema.append(
            {
                "column": cs["column"],
                "top_class": cs["top_class"],
                "classes": [
                    {
                        "cluster_id": int(c["cluster_id"]),
                        "class_name": c["class_name"],
                        "description": c.get("description", ""),
                    }
                    for c in cs["classes"]
                ],
            }
        )
    user_spec = {
        "dataset": dataset,
        "columns": schema,
        "neon_steps": [
            "Identify candidate alignments and object properties between top classes.",
            "For each property, propose rdfs:label, rdfs:comment, domain, range.",
            "Optionally add simple cardinality notes (informal).",
        ],
        "rules": {
            "properties_max": "≤ 12 properties.",
            "json_schema": {
                "object_properties": [
                    {
                        "name": "str",
                        "label": "str",
                        "comment": "str",
                        "domain": "str",
                        "range": "str",
                        "relatedness": "≤ 15 words",
                    }
                ]
            },
        },
        "required_output": ["object_properties"],
    }
    user = (
        "Propose cross-column object properties with domain/range using the top classes per column. "
        "Do not infer instance links.\n\n"
        f"INPUT_JSON:\n{json.dumps(user_spec, ensure_ascii=False)}"
    )
    return system, user


def build_column_population_prompt(
    dataset: str,
    column: str,
    top_class: str,
    classes: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
) -> Tuple[str, str]:
    system = (
        "You are performing NeOn population for ONE column. "
        "Assign each item to ONE class from the provided codebook. "
        "Return STRICT JSON."
    )
    codebook = [
        {
            "cluster_id": int(c["cluster_id"]),
            "class_name": c["class_name"],
            "description": c.get("description", ""),
        }
        for c in classes
    ]
    user_spec = {
        "dataset": dataset,
        "column": column,
        "top_class": top_class,
        "codebook": codebook,
        "items": items,
        "rules": {
            "assignment": "Exactly one cluster_id per item.",
            "json_schema": {"assignments": [{"row": "int", "cluster_id": "int"}]},
        },
        "required_output": ["assignments"],
    }
    user = (
        "Assign each item to exactly one class by cluster_id for THIS column only. "
        "Return JSON with key 'assignments'.\n\n"
        f"INPUT_JSON:\n{json.dumps(user_spec, ensure_ascii=False)}"
    )
    return system, user


# ---------------------------- TTL emission ----------------------------


def emit_ttl_multi(
    dataset: str,
    per_column: List[Dict[str, Any]],
    cross_props: List[Dict[str, Any]],
    per_column_assignments: Dict[str, List[int]],
    per_column_labels: Dict[str, List[str]],
) -> str:
    base = f"http://example.org/{dataset}#"
    ttl: List[str] = []
    ttl += [
        f"@prefix :    <{base}> .",
        "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix owl:  <http://www.w3.org/2002/07/owl#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "",
        ": a owl:Ontology ;",
        f'  rdfs:label "{dataset} Ontology"@en ;',
        '  rdfs:comment "Multi-column ontology induced via LLM-driven NeOn conceptualization and population. Rows are independent."@en .',
        "",
    ]

    for pc in per_column:
        top = pc["top_class"]
        top_desc = pc.get("top_comment", f"Top concept for column {pc['column']}.")
        ttl += [
            f":{top} a owl:Class ;",
            f'  rdfs:label "{ttl_escape(top)}"@en ;',
            f'  rdfs:comment "{ttl_escape(top_desc)}"@en .',
            "",
        ]

    for pc in per_column:
        top = pc["top_class"]
        cid_to_name: Dict[int, str] = {}
        for c in pc["classes"]:
            cid = int(c["cluster_id"])
            cname = to_upper_camel(str(c["class_name"]))
            cid_to_name[cid] = cname
            desc = ttl_escape(c.get("description", f"{cname} concept."))
            ttl += [
                f":{cname} a owl:Class ;",
                f"  rdfs:subClassOf :{top} ;",
                f'  rdfs:label "{ttl_escape(cname)}"@en ;',
                f'  rdfs:comment "{desc}"@en .',
                "",
            ]
        for rel in pc.get("relations", []):
            try:
                s = cid_to_name.get(int(rel["source"]))
                t = cid_to_name.get(int(rel["target"]))
                if not s or not t or s == t:
                    continue
                typ = str(rel.get("type", "related")).lower()
                pred = {
                    "broader": "skos:broader",
                    "narrower": "skos:narrower",
                    "related": "skos:related",
                }.get(typ, "skos:related")
                ttl.append(f":{s} {pred} :{t} .")
            except Exception:
                continue
        ttl.append("")

    if cross_props:
        for p in cross_props:
            name = to_upper_camel(p.get("name", "RelatedTo"))
            label = ttl_escape(p.get("label", name))
            comment = ttl_escape(p.get("comment", ""))
            dom = to_upper_camel(p.get("domain", "Thing"))
            rng = to_upper_camel(p.get("range", "Thing"))
            ttl += [
                f":{name} a owl:ObjectProperty ;",
                f"  rdfs:domain :{dom} ;",
                f"  rdfs:range :{rng} ;",
                f'  rdfs:label "{label}"@en ;',
                f'  rdfs:comment "{comment}"@en .',
                "",
            ]

    ttl.append("# ---- instances per column ----")
    for pc in per_column:
        column = pc["column"]
        cid_to_name = {
            int(c["cluster_id"]): to_upper_camel(str(c["class_name"]))
            for c in pc["classes"]
        }
        labels = per_column_labels[column]
        assigns = per_column_assignments[column]
        for i, (lab, cid) in enumerate(zip(labels, assigns)):
            cname = cid_to_name.get(int(cid), pc["top_class"])
            lid = to_upper_camel(f"{column}_{i}_{lab}")
            ttl += [
                f":{lid} a :{cname} ;",
                f'  rdfs:label "{ttl_escape(str(lab))}"@en .',
            ]
        ttl.append("")
    return "\n".join(ttl)


# ---------------------------- Driver ----------------------------


def _read_single_column_csv(path: str, prefer_col: Optional[str]) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"CSV not found: {path}")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    if df.empty:
        return []
    if prefer_col and prefer_col in df.columns:
        series = df[prefer_col].fillna("").astype(str)
    else:
        series = df[df.columns[0]].fillna("").astype(str)
    return series.tolist()[:10]


def main():
    # Load all columns from separate files
    dataset_name = "separate_files_dataset"
    columns_order = ["subject_name_en", "Job_Title", "Job_Type", "Skill"]
    column_role_hints = ["Subject name", "Job title", "Job type", "Skill list"]

    data_by_column: Dict[str, List[str]] = {}
    for col in columns_order:
        texts = _read_single_column_csv(FILES[col], FILE_COLUMNS.get(col))
        data_by_column[col] = texts
        logging.info(f"Loaded {len(texts)} rows from {FILES[col]} for column '{col}'.")

    cfg = OllamaConfig()

    # Per-column conceptualization
    per_column_results: List[Dict[str, Any]] = []
    with tqdm(
        total=len(columns_order), desc="NeOn conceptualization per column", unit="col"
    ) as pbar:
        for idx, col in enumerate(columns_order):
            role = column_role_hints[idx]
            texts = data_by_column[col]
            sys1, usr1 = build_column_concept_prompt(dataset_name, col, role, texts)
            concept_json = _ollama_chat_json(cfg, sys1, usr1)
            classes = concept_json.get("classes") or []
            relations = concept_json.get("relations") or []
            if not classes or not all(
                "cluster_id" in c and "class_name" in c for c in classes
            ):
                logging.error(
                    f"Invalid 'classes' for column '{col}'. Using minimal fallback."
                )
                classes = [
                    {
                        "cluster_id": 0,
                        "class_name": "Uncategorized",
                        "description": f"Fallback class for {col}",
                    }
                ]
                relations = []
            for c in classes:
                c["class_name"] = to_upper_camel(str(c.get("class_name", "")))
            top_class = to_upper_camel(col)
            top_comment = f"Top concept for column '{col}' with role '{role}'."
            per_column_results.append(
                {
                    "column": col,
                    "top_class": top_class,
                    "top_comment": top_comment,
                    "classes": classes,
                    "relations": relations,
                }
            )
            pbar.update(1)

    # Cross-column mapping
    sys2, usr2 = build_cross_column_mapping_prompt(dataset_name, per_column_results)
    cross_json = _ollama_chat_json(cfg, sys2, usr2)
    object_properties = cross_json.get("object_properties") or []

    # Per-column population
    per_column_assignments: Dict[str, List[int]] = {}
    per_column_labels: Dict[str, List[str]] = {}
    with tqdm(
        total=len(columns_order), desc="NeOn population per column", unit="col"
    ) as pbar:
        for pc in per_column_results:
            col = pc["column"]
            texts = data_by_column[col]
            per_column_labels[col] = texts
            classes = pc["classes"]
            assignments = [0] * len(texts)
            i = 0
            while i < len(texts):
                end = min(i + BATCH_SIZE, len(texts))

                def _snip(s: str) -> str:
                    s = s or ""
                    return (
                        s if len(s) <= TRUNCATE_ITEM_CHARS else s[:TRUNCATE_ITEM_CHARS]
                    )

                batch_items = [
                    {"row": j, "text": _snip(texts[j])} for j in range(i, end)
                ]
                sys3, usr3 = build_column_population_prompt(
                    dataset_name, col, pc["top_class"], classes, batch_items
                )
                pop_json = _ollama_chat_json(cfg, sys3, usr3)
                arr = pop_json.get("assignments") or []
                if not arr or not all("row" in a and "cluster_id" in a for a in arr):
                    logging.error(
                        f"Invalid 'assignments' for column '{col}' rows [{i}:{end}]. Assign all to 0."
                    )
                    for j in range(i, end):
                        assignments[j] = 0
                else:
                    for a in arr:
                        r = int(a["row"])
                        cid = int(a["cluster_id"])
                        if 0 <= r < len(texts):
                            assignments[r] = cid
                i = end
            per_column_assignments[col] = assignments
            pbar.update(1)

    # Emit TTL
    ttl = emit_ttl_multi(
        dataset=dataset_name,
        per_column=per_column_results,
        cross_props=object_properties,
        per_column_assignments=per_column_assignments,
        per_column_labels=per_column_labels,
    )

    out_path = Path(OUT_TTL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ttl, encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
