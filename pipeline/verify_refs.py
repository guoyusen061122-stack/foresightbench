#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参考文献核验：多源交叉 + 标题相似度判定，并**如实记录核验强度**。

沿用上一轮踩坑后定下的三条规矩：
1. **"这个源说没有" ≠ "这些源查不了"**：限流/超时记 `INSUFFICIENT`，不计入判定，
   更不许据此宣判"文献不存在"——那会把基础设施故障变成对作者的学术不端指控。
2. **按独立来源计数**，不把同一来源换个入口算两次。
3. **只有真正核验通过的条目才写进 references.bib**，且 note 里写明确认来源数。
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "lit", "cache")
THRESHOLD = 0.90
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "foresightbench/1.0 (mailto:student@example.edu)"}
_LAST = {}


class Unavailable(Exception):
    """来源当前不可用（限流/超时）——不得当作"未找到"。"""


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def words(s: str) -> set:
    return {w for w in norm(s).split() if len(w) > 2}


def similarity(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    wa, wb = words(a), words(b)
    jac = len(wa & wb) / len(wa | wb) if (wa | wb) else 0.0
    return 0.6 * difflib.SequenceMatcher(None, a, b).ratio() + 0.4 * jac


def _cached(kind: str, url: str, timeout: int = 30):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, kind + "_" +
                     hashlib.sha256(url.encode()).hexdigest()[:20] + ".json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        if d.get("__unavailable__"):
            raise Unavailable(d.get("__reason__", "cached"))
        return d["__body__"]
    gap = 3.0 if kind == "arxiv" else 0.6
    wait = gap - (time.time() - _LAST.get(kind, 0.0))
    if wait > 0:
        time.sleep(wait)
    _LAST[kind] = time.time()
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        json.dump({"__unavailable__": True, "__reason__": f"{type(e).__name__}: {e}"},
                  open(p, "w", encoding="utf-8"))
        raise Unavailable(f"{type(e).__name__}: {e}")
    json.dump({"__body__": body}, open(p, "w", encoding="utf-8"))
    return body


def src_arxiv(title: str):
    url = ("http://export.arxiv.org/api/query?max_results=3&search_query=ti:"
           + urllib.parse.quote('"' + title[:120] + '"'))
    body = _cached("arxiv", url)
    entries = re.findall(r"<entry>(.*?)</entry>", body, re.S)
    best = None
    for e in entries:
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        if not t:
            continue
        cand = re.sub(r"\s+", " ", t.group(1)).strip()
        sim = similarity(title, cand)
        if best is None or sim > best[0]:
            aid = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", e)
            au = re.findall(r"<name>(.*?)</name>", e)
            yr = re.search(r"<published>(\d{4})", e)
            best = (sim, {"source": "arXiv", "title": cand,
                          "arxiv": (aid.group(1) if aid else ""),
                          "authors": au, "year": (yr.group(1) if yr else None)})
    if best and best[0] >= THRESHOLD:
        return best[1]
    return None


def src_crossref(title: str):
    url = ("https://api.crossref.org/works?rows=3&query.bibliographic="
           + urllib.parse.quote(title[:150]) + "&mailto=student@example.edu")
    body = _cached("crossref", url)
    d = json.loads(body)
    best = None
    for it in (d.get("message", {}).get("items") or []):
        cand = (it.get("title") or [""])[0]
        sim = similarity(title, cand)
        if best is None or sim > best[0]:
            au = [f"{a.get('given','')} {a.get('family','')}".strip()
                  for a in (it.get("author") or [])]
            yr = None
            for k in ("published-print", "published-online", "issued"):
                if it.get(k, {}).get("date-parts"):
                    yr = str(it[k]["date-parts"][0][0])
                    break
            best = (sim, {"source": "Crossref", "title": cand,
                          "doi": it.get("DOI", ""), "authors": au, "year": yr,
                          "venue": (it.get("container-title") or [""])[0]})
    if best and best[0] >= THRESHOLD:
        return best[1]
    return None


SOURCES = [("arxiv", src_arxiv), ("crossref", src_crossref)]


def verify(cand: dict) -> dict:
    hits, unavailable = {}, {}
    for name, fn in SOURCES:
        try:
            r = fn(cand["title"])
            if r:
                hits[name] = r
        except Unavailable as e:
            unavailable[name] = str(e)
        except Exception as e:  # noqa: BLE001
            unavailable[name] = f"{type(e).__name__}: {e}"
    res = dict(cand)
    res.update(evidence=hits, sources_unavailable=unavailable,
               n_available=len(hits), n_independent=len(hits))
    if len(hits) >= 2:
        res.update(status="VERIFIED", verified=True)
    elif len(hits) == 1:
        res.update(status="SINGLE_SOURCE", verified=True)
    elif len(unavailable) == len(SOURCES):
        res.update(status="INSUFFICIENT", verified=False)
    else:
        res.update(status="NOT_FOUND", verified=False)
    return res


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=os.path.join(ROOT, "lit", "candidates.json"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    cands = json.load(open(a.candidates, encoding="utf-8-sig"))
    results = []
    for i, c in enumerate(cands, 1):
        r = verify(c)
        results.append(r)
        mark = {"VERIFIED": "OK  ", "SINGLE_SOURCE": "SINGL",
                "NOT_FOUND": "MISS ", "INSUFFICIENT": "N/A  "}[r["status"]]
        print(f"  [{i:>2}/{len(cands)}] {mark} {c['key']:<34} "
              f"{sorted(r['evidence'])}", flush=True)
    out = os.path.join(ROOT, "lit", "refs.json")
    json.dump({"threshold": THRESHOLD, "sources": [n for n, _ in SOURCES],
               "results": results}, open(out, "w", encoding="utf-8", newline="\n"),
              ensure_ascii=False, indent=1)
    ok = [r for r in results if r["verified"]]
    n2 = sum(1 for r in ok if r["n_independent"] >= 2)
    print(f"\n核验通过 {len(ok)}/{len(cands)}（其中 {n2} 条经 2 个独立来源）；"
          f"NOT_FOUND {sum(1 for r in results if r['status']=='NOT_FOUND')} 条")
    if a.apply:
        bib = [os.path.join(ROOT, "references.bib")]
        with open(bib[0], "w", encoding="utf-8", newline="\n") as f:
            f.write("% references.bib —— 由 pipeline/verify_refs.py 自动生成\n")
            f.write(f"% 收录 {len(ok)} 条；独立来源分布："
                    f"2 源 {n2} 条、1 源 {len(ok)-n2} 条\n")
            f.write("% 未通过核验的条目一律不写进来 —— 宁可少引，不可造假。\n\n")
            for r in ok:
                ev = r["evidence"]
                pref = ev.get("arxiv") or ev.get("crossref") or {}
                au = r.get("authors") or pref.get("authors") or []
                auf = (" and ".join(au[:4]) + " and others") if len(au) > 4 \
                    else " and ".join(au)
                doi = r.get("doi") or pref.get("doi") or ""
                arx = pref.get("arxiv") or r.get("arxiv") or ""
                year = r.get("year") or pref.get("year") or ""
                kind = "misc" if arx else "article"
                f.write(f"@{kind}{{{r['key']},\n")
                f.write(f"  title        = {{{r['title']}}},\n")
                if auf:
                    f.write(f"  author       = {{{auf}}},\n")
                if kind == "misc":
                    f.write(f"  howpublished = {{arXiv preprint arXiv:{arx}}},\n")
                elif pref.get("venue"):
                    f.write(f"  journal      = {{{pref['venue']}}},\n")
                if year:
                    f.write(f"  year         = {{{year}}},\n")
                if doi:
                    f.write(f"  doi          = {{{doi}}},\n")
                f.write(f"  note         = {{{r['n_independent']} independent "
                        f"source(s): {', '.join(sorted(ev))}}},\n")
                f.write("}\n\n")
        print(f"已写出 references.bib（{len(ok)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
