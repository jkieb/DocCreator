# -*- coding: utf-8 -*-
# Voraussetzungen:
#   pip install python-dotenv openai
# Env (.env):
#   OPENAI_API_KEY=sk-...
# Optional:
#   OPENAI_MODEL=gpt-4o-mini

import os, sys, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if len(sys.argv) < 2:
    print("Verwendung: python segment_only.py <pfad_zur_datei.py>")
    sys.exit(1)

pfad = sys.argv[1]
with open(pfad, "r", encoding="utf-8") as f:
    code = f.read()

# Hinweis an das Modell: Nur segmentieren, keine Erklärungen; aber mit Code statt Zeilennummern
system_msg = (
    "Du segmentierst Python-Code in LOGISCHE EINHEITEN. "
    "Gib ausschließlich JSON zurück mit dem Key 'segments'. "
    "Jedes Segment hat: title (kurz), code (genauer Textblock als String), rationale (1 Satz). "
    "Segmentiere grob nach Imports, Konfiguration, Datenmodelle, Funktionen/Klassen, Main-Block, I/O, Hilfsblöcke. "
    "Keine zusätzlichen Texte, keine Formatierung außerhalb des JSON."
)

user_msg = (
    "Segmentiere NUR. Analysiere NICHT jede Zeile. "
    "Gib JSON gemäß Vorgabe zurück.\n\n"
    f"```python\n{code}\n```"
)

client = OpenAI()
resp = client.chat.completions.create(
    model=model,
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ],
    temperature=0.1,
)

# Robust auslesen und schön drucken
inhalt = resp.choices[0].message.content
try:
    data = json.loads(inhalt)
except json.JSONDecodeError:
    # Falls das Modell wider Erwarten Text liefert, roh ausgeben
    print(inhalt)
    sys.exit(0)

print(json.dumps(data, ensure_ascii=False, indent=2))


