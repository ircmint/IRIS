"""
irc_kb_builder.py
-----------------
Parses IRC35_2015.pdf and IRC67_2022.pdf SEPARATELY (each PDF produces its own
independent knowledge base JSON — they are never merged) and extracts clause
level text ONLY for the sections requested by the user:

    IRC:35-2015 -> Sections 3, 4, 6.1, 6.2, 7, 8, 11
    IRC:67-2022 -> Sections 3, 11, 13, 14, 15, 16, 17, 24, 25, 26

Each clause becomes a record:
    {
        "irc_code": "IRC35_2015",
        "section": "4",
        "clause_id": "4.6",
        "heading": "Longitudinal Marking for Undivided Roads",
        "text": "<verbatim clause text pulled from the pdf>",
        "page": 23
    }

Run standalone:
    python3 irc_kb_builder.py
"""

import json
import re
import os
import pdfplumber

from config import IRC_DOCS, KB_DIR


CLAUSE_RE = re.compile(r"(?m)^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-Z][A-Za-z0-9 ,\-/&\.\(\)']{2,90})\s*$")
INLINE_CLAUSE_RE = re.compile(
    r"(?m)^\s*(\d{1,2}\.\d{1,2})\s+"
)


def _requested_top_sections(requested):
    """Return the set of top-level section numbers implied by the requested list
    (e.g. '6.1' implies top-level section 6)."""
    tops = set()
    for r in requested:
        tops.add(int(r.split(".")[0]))
    return tops


def _extract_pages_text(pdf, start, end):
    pages_text = []
    for i in range(start, min(end, len(pdf.pages))):
        t = pdf.pages[i].extract_text() or ""

        if i == 29:   # page where clause 4.7 is located
            print("=" * 80)
            print("RAW PAGE 29")
            print("=" * 80)
            print(t)
            print("=" * 80)

        pages_text.append((i, t))
    return pages_text


def _split_into_clauses(irc_code, section_num, pages_text):
    """Within a section's raw page text, split on numbered clause headings like
    '4.6 Longitudinal Marking ...' and attach page numbers."""
    records = []
    # Flatten to a single string but keep a page marker we can map back from
    combined = ""
    page_offsets = []  # (char_offset, page_index)
    for page_idx, txt in pages_text:
        page_offsets.append((len(combined), page_idx))
        combined += txt + "\n"

    matches = list(INLINE_CLAUSE_RE.finditer(combined))
    if not matches:
        # no sub-clause numbering found; keep whole section as one record
        records.append({
            "irc_code": irc_code,
            "section": str(section_num),
            "clause_id": str(section_num),
            "heading": "",
            "text": combined.strip()[:4000],
            "page": pages_text[0][0] if pages_text else None,
        })
        return records

    for idx, m in enumerate(matches):
        clause_id = m.group(1)
        # only keep clauses that belong to this top-level section
        if int(clause_id.split(".")[0]) != int(section_num):
            continue
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(combined)
        chunk = combined[start:end].strip()
        if clause_id == "4.7":
            print("=" * 80)
            print(chunk)
            print("=" * 80)
        # heading = rest of the first line after the clause number
        first_line = chunk.split("\n", 1)[0]
        heading = first_line[len(clause_id):].strip(" .:-")
        # find page for this offset
        page_no = None
        for off, pidx in page_offsets:
            if off <= start:
                page_no = pidx
        records.append({
            "irc_code": irc_code,
            "section": str(section_num),
            "clause_id": clause_id,
            "heading": heading[:120],
            "text": chunk[:4000],
            "page": page_no,
        })
    return records


def build_kb_for_doc(doc_key, doc_cfg):
    pdf_path = doc_cfg["pdf_path"]
    section_pages = doc_cfg["section_pages"]
    requested = doc_cfg["requested_sections"]
    top_sections = _requested_top_sections(requested)

    all_records = []
    with pdfplumber.open(pdf_path) as pdf:
        for sec_num in sorted(top_sections):
            if sec_num not in section_pages:
                print(f"[WARN] {doc_key}: requested section {sec_num} not in section_pages map, skipping")
                continue
            start, end = section_pages[sec_num]
            pages_text = _extract_pages_text(pdf, start, end)
            recs = _split_into_clauses(doc_key, sec_num, pages_text)
            all_records.extend(recs)

    # Filter down to exactly the requested clause_ids where a specific
    # sub-clause (e.g. "6.1") was asked for, otherwise keep the whole section.
    exact_subclauses = {r for r in requested if "." in r}
    whole_sections = {r for r in requested if "." not in r}

    final_records = []
    for rec in all_records:
        cid = rec["clause_id"]
        top = cid.split(".")[0]
        if top in whole_sections:
            final_records.append(rec)
        elif any(cid == s or cid.startswith(s + ".") for s in exact_subclauses):
            final_records.append(rec)

    out_path = os.path.join(KB_DIR, f"{doc_key}_kb.json")
    with open(out_path, "w") as f:
        json.dump(final_records, f, indent=2)
    print(f"[OK] {doc_key}: extracted {len(final_records)} clause records -> {out_path}")
    return final_records


def build_all():
    results = {}

    for doc_key, doc_cfg in IRC_DOCS.items():
        results[doc_key] = build_kb_for_doc(doc_key, doc_cfg)

    print("\n")
    print("=" * 90)
    print(f"{'IRC Document':15} {'Sections':35} {'#Sections':10} {'#Clauses':10}")
    print("=" * 90)

    for doc, cfg in IRC_DOCS.items():
        print(
            f"{doc:15}"
            f"{', '.join(cfg['requested_sections']):35}"
            f"{len(cfg['requested_sections']):10}"
            f"{len(results[doc]):10}"
        )

    print("=" * 90)

    return results


if __name__ == "__main__":
    build_all()
