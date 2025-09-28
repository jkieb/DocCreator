# -*- coding: utf-8 -*-
# Linear: 1) Segmentiert Python-Code via ChatGPT  2) Erklärt Segmente  3) Schreibt Markdown.
# Nutzung:
#   export OPENAI_API_KEY=sk-...
#   python doccreator.py doccreator.py output.md
#
# Abhängigkeiten:
#   pip install openai python-dotenv

import os, sys, json, time, textwrap
from dotenv import load_dotenv
from openai import OpenAI

# -------------------- Setup --------------------
load_dotenv()
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

if len(sys.argv) < 3:
    print("Verwendung: python doccreator.py <pfad_zur_datei.py> <output.md>")
    sys.exit(1)

pfad_code = sys.argv[1]
pfad_md   = sys.argv[2]

with open(pfad_code, "r", encoding="utf-8") as f:
    full_code = f.read()

client = OpenAI()

# -------------------- Schritt 1: Segmentieren --------------------
system_msg_seg = (
    "Du segmentierst Python-Code in LOGISCHE EINHEITEN. "
    "Gib ausschließlich JSON zurück mit dem Key 'segments'. "
    "Jedes Segment hat: title (kurz), code (genauer Textblock als String), rationale (1 Satz). "
    "Segmentiere grob nach Imports, Konfiguration, Datenmodelle, Funktionen/Klassen, Main-Block, I/O, Hilfsblöcke. "
    "Keine zusätzlichen Texte, keine Formatierung außerhalb des JSON."
)

user_msg_seg = (
    "Segmentiere NUR. Analysiere NICHT jede Zeile. Gib JSON gemäß Vorgabe zurück.\n\n"
    f"```python\n{full_code}\n```"
)

resp_seg = client.chat.completions.create(
    model=model,
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": system_msg_seg},
        {"role": "user", "content": user_msg_seg},
    ],
    temperature=0.1,
)
seg_json_text = resp_seg.choices[0].message.content
seg_data = json.loads(seg_json_text)  # bewusst ohne try/except

segments = []
for item in seg_data.get("segments", []):
    c = (item.get("code") or "").strip()
    if c:
        segments.append({
            "title": item.get("title") or "Ohne Titel",
            "code": c,
            "rationale": item.get("rationale") or ""
        })

# -------------------- Markdown: Header + TOC --------------------
doc_lines = []
doc_lines.append("# Erklärung des Codes\n")
doc_lines.append(f"Dieses Dokument erklärt {len(segments)} Segmente. Jedes Kapitel enthält eine kurze Erklärung, Besonderheiten, Hinweise und Schnittstellen.\n")
doc_lines.append("## Inhalt\n")
for i, seg in enumerate(segments, 1):
    # Slug inline ohne Funktion
    t = seg["title"]
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in t)
    slug = "-".join(s for s in slug.split("-") if s)[:80] or "abschnitt"
    doc_lines.append(f"- [{i}. {t}](#{i}-{slug})")
doc_lines.append("")

# -------------------- Schritt 2: Erklären pro Segment --------------------
system_msg_explain = (
    "Du bist ein präziser Senior-Developer und Tech-Writer. "
    "Erkläre kompakt, korrekt, mit klaren Bullet Points."
)

for i, seg in enumerate(segments, 1):
    user_msg_explain = textwrap.dedent(f"""
        Erkläre prägnant und technisch korrekt dieses Code-Segment.
        Sprache: Deutsch.
        Kontext:
        - Segment-Nummer: {i}
        - Titel: {seg['title']}
        - Kurzbegründung: {seg['rationale'] or "—"}

        Aufgaben:
        1) 4–8 Sätze zu Zweck und Ablauf.
        2) Besonderheiten/Randfälle (max. 6 Bullet Points).
        3) Hinweise (Performance/Sicherheit/Wartung, max. 4 Bullet Points).
        4) Schnittstellen zu anderen Segmenten (Input/Output), falls erkennbar.

        Gib NUR den folgenden Markdown-Block zurück:
        ### {i}. {seg['title']}
        ```python
        <kurzer relevanter Codeauszug oder Signatur, falls sinnvoll>
        ```
        **Erklärung**
        - ...
        **Besonderheiten & Randfälle**
        - ...
        **Hinweise**
        - ...
        **Schnittstellen**
        - ...

        Vollständiger Segment-Code:
        --- CODE START ---
        {seg['code']}
        --- CODE END ---
    """).strip()

    resp_exp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=900,
        messages=[
            {"role": "system", "content": system_msg_explain},
            {"role": "user", "content": user_msg_explain},
        ],
    )
    block = resp_exp.choices[0].message.content.strip()
    anchor = f"### {i}. {seg['title']}"
    if not block.lstrip().startswith(anchor):
        block = anchor + "\n\n" + block
    doc_lines.append(block)
    doc_lines.append("")

# -------------------- Footer + Schreiben --------------------
footer = f"---\n*Generiert am:* {time.strftime('%Y-%m-%d %H:%M:%S')}  \n*Modell:* {model} • *Temp:* {temperature} • *Segmente:* {len(segments)}"
doc_lines.append(footer)
md = "\n".join(doc_lines)

with open(pfad_md, "w", encoding="utf-8") as f:
    f.write(md)

print(f"✅ Fertig: {pfad_md}")
