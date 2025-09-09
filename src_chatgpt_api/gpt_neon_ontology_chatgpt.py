#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeOn-style ontology induction from ONE CSV at a time using the ChatGPT API.
- Generation & population: OPENAI_MODEL_GEN (default: gpt-4o-mini)
- Analyst: OPENAI_MODEL_ANALYST (default: gpt-4o)
- Config via a single .env (dotenv). CLI can override.

Minimal .env:
  OPENAI_API_KEY=<your key>

Optional .env:
  OPENAI_BASE_URL=https://api.openai.com/v1
  OPENAI_ORG=<org id>
  OPENAI_PROJECT=<project id>
  OPENAI_MODEL_GEN=gpt-4o-mini
  OPENAI_MODEL_ANALYST=gpt-4o
  OPENAI_TIMEOUT=120

Usage:
  python gpt_neon_ontology_chatgpt.py \
    --input dataset/cleaned_Subject.csv \
    --module Subject \
    --dataset ComputerEducationData \
    --out ./output/ontology_subject.ttl \
    --analysis_out ./output/ontology_subject.analysis.json \
    --analysis_md  ./output/ontology_subject.analysis.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

# OpenAI SDK v1.x
try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit(
        "Missing dependencies. Install: pip install openai python-dotenv tqdm"
    ) from e

# ---------------------------- Defaults ----------------------------

# Load .env early so getenv works for defaults
load_dotenv()

DEFAULT_MODEL_GEN = os.getenv("OPENAI_MODEL_GEN", "gpt-4o-mini")
DEFAULT_MODEL_ANALYST = os.getenv("OPENAI_MODEL_ANALYST", "gpt-4o")
DEFAULT_TEMPERATURE_GEN = 0.1
DEFAULT_TEMPERATURE_ANALYST = 0.1
NUM_CTX = 128000  # depends on model
MAX_TOKENS = 4096
DEFAULT_BATCH = 200
TRUNCATE_ITEM_CHARS = 160
MIN_BATCH = 12

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------- Client ----------------------------


@dataclass
class ChatCfg:
    model: str
    temperature: float = 0.1
    timeout: int = int(os.getenv("OPENAI_TIMEOUT", "120"))


def build_client() -> OpenAI:
    base = os.getenv("OPENAI_BASE_URL") or None
    api_key = os.getenv("OPENAI_API_KEY")
    org = os.getenv("OPENAI_ORG") or None
    project = os.getenv("OPENAI_PROJECT") or None
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Create .env with OPENAI_API_KEY=<your key>"
        )

    # openai>=1.0.0
    if base:
        return OpenAI(api_key=api_key, base_url=base, organization=org, project=project)
    return OpenAI(api_key=api_key, organization=org, project=project)


def chat_json(
    client: OpenAI, cfg: ChatCfg, system_prompt: str, user_prompt: str, retries: int = 3
) -> str:
    """Ask for JSON only using response_format. Returns raw text."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                temperature=max(cfg.temperature * (0.6**attempt), 0.05),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=MAX_TOKENS,
                timeout=cfg.timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    logging.warning(f"chat_json failed after {retries} attempts: {last_err}")
    return ""


def chat_text(
    client: OpenAI, cfg: ChatCfg, system_prompt: str, user_prompt: str, retries: int = 3
) -> str:
    """Ask for plain text (used for TSV fallback). Returns raw text."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                temperature=max(cfg.temperature * (0.6**attempt), 0.05),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                timeout=cfg.timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    logging.warning(f"chat_text failed after {retries} attempts: {last_err}")
    return ""


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


# ---------------------------- Prompts ----------------------------


def build_module_concept_prompts(
    dataset: str, module: str, role_hint: str, examples: List[str]
) -> Tuple[str, str]:
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
    user = (
        "Respond with JSON only. No prose.\n\n"
        "[OUTPUT_FORMAT]\n"
        '{"top_class": str, "classes": [{"cluster_id": int, "class_name": str, "description": str}], '
        '"relations": [{"source": str, "target": str, "type": "broader|narrower|related"}]}\n'
        "[/OUTPUT_FORMAT]\n"
        f"INPUT_JSON:\n{json.dumps(user_spec, ensure_ascii=False)}"
    )
    return system, user


def build_population_prompts(
    dataset: str,
    module: str,
    top_class: str,
    classes: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
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
    system_json = "You return STRICT JSON only. No commentary."
    user_json = (
        "Assign each item to exactly one class by cluster_id for THIS module only.\n"
        "[OUTPUT_FORMAT]\n"
        '{"assignments": [{"row": int, "cluster_id": int}]}\n'
        "[/OUTPUT_FORMAT]\n"
        f"INPUT_JSON:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    system_tsv = "You output only lines 'row<TAB>cluster_id'. No JSON. No extra text."
    user_tsv = (
        "Example:\n0\t2\n1\t0\n2\t1\n"
        f"Classes: {json.dumps(class_min, ensure_ascii=False)}\n"
        f"Items: {json.dumps(items, ensure_ascii=False)}\n"
        "Output now."
    )
    return (system_json, user_json), (system_tsv, user_tsv)


def build_analysis_prompts(dataset: str, module: str, ttl_text: str) -> Tuple[str, str]:
    system = (
        "You are an ontology QA assistant. Please analyze the following ontology (TTL format).\n"
        "Output language: Thai.\n\n"
        "Your report must contain ONLY the fixed sections below (Markdown in Thai):\n"
        "1.แนวคิด (concept)\n"
        "- แสดง class ทั้งหมด โดยใช้ชื่อจริงจาก rdfs:label เท่านั้น\n"
        "- แสดง subclass hierarchy ถ้ามี โดยใช้ชื่อจาก rdfs:label\n"
        "- วิเคราะห์ว่ามี class ที่ซ้ำซ้อนหรือไม่\n\n"
        "2.คุณลักษณะ (property)\n"
        "- ระบุ property ทั้งหมดที่พบ โดยใช้ชื่อจริงจาก rdfs:label\n"
        "- ระบุ domain และ range ของแต่ละ property\n"
        "- แบ่งเป็น object property กับ datatype property\n"
        "- ระบุคุณสมบัติพิเศษของ property ถ้ามี (owl:Functional, owl:InverseFunctional, owl:Transitive ฯลฯ)\n\n"
        "3.ความสัมพันธ์ (relationship)\n"
        "3.1) ความสัมพันธ์แบบลำดับชั้น (subclass): คู่ ChildClass – ParentClass โดยใช้ชื่อจาก rdfs:label\n"
        "3.2) ความสัมพันธ์แบบเป็นส่วนหนึ่ง (part-of): คู่ Part – Whole โดยใช้ชื่อจาก rdfs:label\n"
        "3.3) ความสัมพันธ์เชิงความหมาย (syn-of): คู่ Synonym – CanonicalTerm โดยใช้ชื่อจาก rdfs:label\n"
        "3.4) ความสัมพันธ์การเป็นตัวแทน (instance-of): ระบุ instance และ class โดยใช้ชื่อจริงจาก rdfs:label (เช่น skill จาก CSV) ไม่ใช้ IRI เช่น Skill1, Skill2\n\n"
        "4.ข้อกำหนดในการสร้างความสัมพันธ์ (axiom)\n"
        "- ระบุ axioms ที่ปรากฏ เช่น domain/range restrictions, disjointness, equivalence, transitivity, symmetry\n"
        "- ถ้าไม่มี ให้เสนอ axioms ที่ควรมีเพื่อทำให้ ontology แข็งแรงขึ้น\n\n"
        "Additional requirements:\n"
        "- ใช้ชื่อจริงจาก rdfs:label สำหรับการรายงานทุก class, property, instance และ relationship\n"
        "- ตรวจสอบ ความครอบคลุม (coverage): instance ทุกตัวถูก assign เข้าสู่ class หรือไม่\n"
        "- ตรวจสอบ คุณภาพการตั้งชื่อ (naming quality): class/property/instance ตั้งชื่อตามมาตรฐานหรือไม่ (UpperCamelCase, ไม่มีช่องว่าง, ภาษาเดียว)\n"
        "- วิเคราะห์ consistency: มี class หรือ property ที่ขัดแย้งกันหรือไม่\n"
        "- วิเคราะห์ redundancy: มี class/property ที่ซ้ำซ้อนโดยไม่จำเป็นหรือไม่\n"
        "- ให้ recommendations: ควรเพิ่ม property/axiom/comment หรือลดความซ้ำซ้อนตรงไหน\n"
        "- Output เป็น JSON object ที่มีฟิลด์:\n"
        '  - "markdown": str (รายงาน Markdown ภาษาไทยตามหัวข้อด้านบน)\n'
        '  - "concepts": [str]\n'
        '  - "properties": [{"name": str, "type": "object|datatype", "domain": str, "range": str, "owl_props": [str]}]\n'
        '  - "relationships": {"subclass": [[str,str]], "part_of": [[str,str]], "syn_of": [[str,str]], "instance_of": [[str,str]]}\n'
        '  - "axioms": [str]\n'
        '  - "issues": [str]\n'
        '  - "coverage": [str]\n'
        '  - "naming_issues": [str]\n'
        '  - "redundancy": [str]\n'
        '  - "recommendations": [str]\n'
    )
    schema = {
        "dataset": dataset,
        "module": module,
        "rules": [
            "Infer subclass from rdfs:subClassOf.",
            "Infer instance-of from rdf:type of subclasses.",
            "Detect coverage issues: instances not mapped to any class.",
            "Detect naming issues: class/property not in UpperCamelCase or with invalid chars.",
            "Use rdfs:label as the display name for all entities, not IRI.",
            "Propose part-of only if composition implied by labels/comments/SKOS.",
            "Propose syn-of only for clear synonymy.",
            "List properties if any custom predicates exist; otherwise suggest needed ones.",
            "Propose axioms like disjointness, transitivity, symmetry, domain/range.",
            "Always include recommendations for ontology improvement.",
        ],
        "ttl_excerpt": ttl_text[:12000],
    }
    user = (
        "Respond with JSON only. No prose.\n"
        "[OUTPUT_FORMAT]\n"
        "{\n"
        '  "markdown": str,\n'
        '  "concepts": [str],\n'
        '  "properties": [{"name": str, "type": "object|datatype", "domain": str, "range": str, "owl_props": [str]}],\n'
        '  "relationships": {"subclass": [[str,str]], "part_of": [[str,str]], "syn_of": [[str,str]], "instance_of": [[str,str]]},\n'
        '  "axioms": [str],\n'
        '  "issues": [str],\n'
        '  "coverage": [str],\n'
        '  "naming_issues": [str],\n'
        '  "redundancy": [str],\n'
        '  "recommendations": [str]\n'
        "}\n"
        "[/OUTPUT_FORMAT]\n"
        f"INPUT_JSON:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    return system, user


# ---------------------------- Parsers ----------------------------


def parse_assignments_json(text: str) -> Optional[List[Dict[str, int]]]:
    if not text:
        return None
    try:
        data = json.loads(text)
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
        f':{M} a owl:Class ; rdfs:label "{M}"@en .',
        "",
    ]

    for c in classes:
        name = to_upper_camel(str(c["class_name"]))
        desc = (c.get("description") or "").replace('"', '\\"')
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
        safe_text = (text or "").replace('"', '\\"')
        ttl += [
            f":{label} a :{M} ;",
            f'  rdfs:label "{safe_text}"@en .',
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


# ---------------------------- Main ----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--module", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--column", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--analysis_out", default=None, type=Path, help="Write JSON QA report"
    )
    ap.add_argument(
        "--analysis_md", default=None, type=Path, help="Write Markdown QA summary"
    )
    ap.add_argument("--model_gen", default=DEFAULT_MODEL_GEN)
    ap.add_argument("--model_analyst", default=DEFAULT_MODEL_ANALYST)
    ap.add_argument("--temperature_gen", type=float, default=DEFAULT_TEMPERATURE_GEN)
    ap.add_argument(
        "--temperature_analyst", type=float, default=DEFAULT_TEMPERATURE_ANALYST
    )
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--test_rows", type=int, default=None)
    args = ap.parse_args()

    client = build_client()
    cfg_gen = ChatCfg(model=args.model_gen, temperature=args.temperature_gen)
    cfg_analyst = ChatCfg(
        model=args.model_analyst, temperature=args.temperature_analyst
    )

    items = _read_single_column_csv(args.input, args.column)
    if args.test_rows is not None:
        items = items[: args.test_rows]
    logging.info(f"Loaded {len(items)} rows from {args.input}")

    # ---- Concept phase
    role_hint = "Skill list" if "skill" in args.module.lower() else "Subject name"
    sys_c, usr_c = build_module_concept_prompts(
        args.dataset, args.module, role_hint, items
    )
    raw_concept = chat_json(client, cfg_gen, sys_c, usr_c, retries=4)
    concept = {}
    try:
        concept = json.loads(raw_concept) if raw_concept else {}
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

    # ---- Population with adaptive batching and TSV fallback
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

            (sys_j, usr_j), (sys_t, usr_t) = build_population_prompts(
                args.dataset, args.module, top_class, classes, batch
            )

            raw_json = chat_json(client, cfg_gen, sys_j, usr_j, retries=4)
            parsed = parse_assignments_json(raw_json)

            if parsed is None:
                raw_tsv = chat_text(client, cfg_gen, sys_t, usr_t, retries=3)
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

    # ---- Render TTL
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

    # ---- Final analysis (Thai headings) with analyst model
    if args.analysis_out or args.analysis_md:
        sys_a, usr_a = build_analysis_prompts(args.dataset, args.module, ttl)
        raw_analysis = chat_json(client, cfg_analyst, sys_a, usr_a, retries=4)
        report = {}
        try:
            report = json.loads(raw_analysis) if raw_analysis else {}
        except Exception:
            logging.error("Analysis JSON parse failed. Writing minimal Thai markdown.")
            report = {"markdown": "# รายงานวิเคราะห์ออนโทโลยี\n\n(ไม่สามารถวิเคราะห์ได้)"}

        if args.analysis_out:
            args.analysis_out.parent.mkdir(parents=True, exist_ok=True)
            args.analysis_out.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(str(args.analysis_out))

        if args.analysis_md:
            md = report.get("markdown")
            if not md:
                # Compose Markdown from fields in Thai
                def _mk_pairs(title, pairs):
                    if not pairs:
                        return f"### {title}\n- ไม่มี\n"
                    return (
                        "### "
                        + title
                        + "\n"
                        + "\n".join(
                            f"- {json.dumps(x, ensure_ascii=False)}" for x in pairs
                        )
                        + "\n"
                    )

                parts = []
                parts.append(
                    f"# รายงานวิเคราะห์ออนโทโลยี: {args.dataset} {to_upper_camel(args.module)}\n"
                )
                parts.append(
                    "## 1.แนวคิด (concept)\n"
                    + "\n".join(f"- {c}" for c in (report.get("concepts") or []))
                    + "\n"
                )
                props = report.get("properties") or []
                parts.append(
                    "## 2.คุณลักษณะ (property)\n"
                    + (
                        "\n".join(
                            f"- {json.dumps(p, ensure_ascii=False)}" for p in props
                        )
                        if props
                        else "- ไม่มี"
                    )
                    + "\n"
                )
                rel = report.get("relationships", {}) or {}
                parts.append("## 3.ความสัมพันธ์ (relationship)\n")
                parts.append(
                    _mk_pairs("3.1) ความสัมพันธ์แบบลำดับชั้น (subclass)", rel.get("subclass"))
                )
                parts.append(
                    _mk_pairs(
                        "3.2) ความสัมพันธ์แบบเป็นส่วนหนึ่ง (part-of)", rel.get("part_of")
                    )
                )
                parts.append(
                    _mk_pairs("3.3) ความสัมพันธ์เชิงความหมาย (syn-of)", rel.get("syn_of"))
                )
                parts.append(
                    _mk_pairs(
                        "3.4) ความสัมพันธ์การเป็นตัวแทน (instance-of)",
                        rel.get("instance_of"),
                    )
                )
                axioms = report.get("axioms") or []
                parts.append(
                    "## 4.ข้อกำหนดในการสร้างความสัมพันธ์ (axiom)\n"
                    + ("\n".join(f"- {a}" for a in axioms) if axioms else "- ไม่มี")
                    + "\n"
                )
                md = "\n".join(parts)

            args.analysis_md.parent.mkdir(parents=True, exist_ok=True)
            args.analysis_md.write_text(md, encoding="utf-8")
            print(str(args.analysis_md))


if __name__ == "__main__":
    main()
