# Erklärung des Codes

Dieses Dokument erklärt 17 Segmente. Jedes Kapitel enthält eine kurze Erklärung, Besonderheiten, Hinweise und Schnittstellen.

## Inhalt

- [1. Imports](#1-imports)
- [2. Model Configuration](#2-model-configuration)
- [3. Load Backbone Function](#3-load-backbone-function)
- [4. Dataset Class](#4-dataset-class)
- [5. Load Cached DataFrame Function](#5-load-cached-dataframe-function)
- [6. Streamlit UI Setup](#6-streamlit-ui-setup)
- [7. Add Customer Logic](#7-add-customer-logic)
- [8. Upload Table Logic](#8-upload-table-logic)
- [9. Load CSV Logic](#9-load-csv-logic)
- [10. Temporary Info Message](#10-temporary-info-message)
- [11. Display Data Logic](#11-display-data-logic)
- [12. Training Parameters Setup](#12-training-parameters-setup)
- [13. Data Preparation](#13-data-preparation)
- [14. Model Training Logic](#14-model-training-logic)
- [15. Evaluation Logic](#15-evaluation-logic)
- [16. Classification Report Logic](#16-classification-report-logic)
- [17. Download Functionality](#17-download-functionality)

### 1. Imports
```python
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from torch import nn
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
```
**Erklärung**
- Dieses Segment importiert essentielle Bibliotheken und Module, die für die Funktionalität der Anwendung erforderlich sind.
- Die Imports decken verschiedene Bereiche ab, darunter Datenverarbeitung (Pandas, NumPy), maschinelles Lernen (Torch, Sentence Transformers, Scikit-Learn) und Webanwendungen (Streamlit).
- Jedes Modul wird für spezifische Aufgaben verwendet, wie z.B. Datenmanipulation, Modelltraining und -bewertung sowie die Erstellung von Benutzeroberflächen.
- Die Verwendung dieser Bibliotheken ermöglicht eine modulare und wartbare Struktur der Anwendung.

**Besonderheiten & Randfälle**
- Die Importreihenfolge kann die Ausführung beeinflussen, insbesondere bei Abhängigkeiten zwischen Modulen.
- Einige Module (z.B. Torch) benötigen spezifische Hardware (GPU) für optimale Leistung.
- Bei fehlenden Modulen kann es zu ImportError kommen, was die Anwendung zum Absturz bringen kann.
- Versionskonflikte zwischen Bibliotheken können unerwartete Fehler verursachen.
- Bestimmte Module (wie Streamlit) erfordern eine spezifische Umgebung (z.B. Webserver), um korrekt zu funktionieren.
- Die Verwendung von `os` kann plattformabhängige Probleme verursachen, wenn Pfade nicht korrekt behandelt werden.

**Hinweise**
- Achten Sie darauf, die Bibliotheken regelmäßig zu aktualisieren, um Sicherheitslücken zu schließen.
- Verwenden Sie virtuelle Umgebungen, um Abhängigkeiten zu isolieren und Konflikte zu vermeiden.
- Überprüfen Sie die Dokumentation der Bibliotheken auf Änderungen in der API, die die Wartung beeinflussen könnten.
- Optimieren Sie die Importanweisungen, um nur benötigte Module zu laden und die Startzeit der Anwendung zu verkürzen.

**Schnittstellen**
- Die importierten Module stellen Funktionen und Klassen bereit, die in anderen Segmenten zur Datenverarbeitung, Modelltraining und Benutzerinteraktion verwendet werden.
- Beispielsweise wird `pandas` für die Datenmanipulation und `torch` für das maschinelle Lernen in nachfolgenden Segmenten benötigt.

### 2. Model Configuration
```python
class Model(nn.Module):
    def __init__(self, backbone: SentenceTransformer, hidden_dim: int, n_products: int, n_prios: int):
        super().__init__()
        self.backbone   = backbone
        emb_dim         = backbone.get_sentence_embedding_dimension()
        self.shared     = nn.Linear(emb_dim, hidden_dim)
        self.product_head = nn.Linear(hidden_dim, n_products)
        self.prio_head  = nn.Linear(hidden_dim, n_prios)

    def forward(self, features, task_id: int):
        if isinstance(features, torch.Tensor):
            embeddings = features
        elif isinstance(features, (list, str)):
            embeddings = self.backbone.encode(features, convert_to_tensor=True)
        else:
            output = self.backbone(features)
            if isinstance(output, dict):
                embeddings = output.get(
                    'sentence_embedding',
                    next(iter(output.values()))
                )
            else:
                embeddings = output
        h = torch.relu(self.shared(embeddings))
        return self.product_head(h) if task_id == 0 else self.prio_head(h)
```
**Erklärung**
- Dieses Code-Segment definiert ein neuronales Netzwerk-Modell in PyTorch, das auf einem vortrainierten SentenceTransformer basiert.
- Im Konstruktor (`__init__`) werden die Netzwerkarchitektur und die erforderlichen Schichten initialisiert, einschließlich einer gemeinsamen Schicht und zwei Ausgabeschichten für Produkte und Prioritäten.
- Die `forward`-Methode verarbeitet Eingabedaten, die entweder als Tensor, Liste oder String vorliegen können, und gibt die entsprechenden Vorhersagen basierend auf dem `task_id` zurück.

**Besonderheiten & Randfälle**
- Unterstützung für verschiedene Eingabetypen (Tensor, Liste, String).
- Verwendung von `torch.relu` zur Aktivierung der gemeinsamen Schicht.
- Dynamische Auswahl der Ausgabeschicht basierend auf `task_id`.
- Möglichkeit, dass die `backbone`-Ausgabe ein Dictionary ist, was zusätzliche Flexibilität bietet.
- Fehlerbehandlung für unerwartete Eingabetypen ist nicht implementiert.
- Abhängigkeit von der korrekten Dimensionierung der Eingabedaten.

**Hinweise**
- Achten Sie auf die Dimensionen der Eingabedaten, um Dimensionierungsfehler zu vermeiden.
- Bei der Verwendung von `SentenceTransformer` sollte sichergestellt werden, dass das Modell korrekt geladen ist.
- Die Performance kann durch Batch-Verarbeitung der Eingabedaten verbessert werden.
- Regelmäßige Wartung und Updates des `backbone`-Modells sind empfehlenswert, um die Genauigkeit zu gewährleisten.

**Schnittstellen**
- **Input:** `features` (Tensor, Liste oder String) und `task_id` (int).
- **Output:** Vorhersageergebnisse aus `product_head` oder `prio_head`, abhängig von `task_id`.

### 3. Load Backbone Function
```python
@st.cache_resource(show_spinner="Lade Basis-Modell...")
def load_backbone(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    return SentenceTransformer(model_name)
```
**Erklärung**
- Die Funktion `load_backbone` lädt ein vortrainiertes Modell der Klasse `SentenceTransformer`, das für die Verarbeitung von Texten verwendet wird.
- Durch die Verwendung von `@st.cache_resource` wird das Modell im Cache gespeichert, um die Ladezeiten bei wiederholten Aufrufen zu reduzieren.
- Der Parameter `model_name` ermöglicht es, verschiedene Modelle zu laden, wobei der Standardwert auf ein multilinguales Modell gesetzt ist.
- Der Spinner zeigt dem Benutzer an, dass das Modell geladen wird, was die Benutzererfahrung verbessert.

**Besonderheiten & Randfälle**
- Das Caching funktioniert nur, wenn die Funktion mit denselben Parametern aufgerufen wird.
- Bei ungültigen Modellnamen wird eine Ausnahme ausgelöst, die behandelt werden sollte.
- Die Funktion ist nicht thread-sicher; parallele Aufrufe könnten zu unerwartetem Verhalten führen.
- Der Cache kann bei Änderungen am Modell oder den Parametern ungültig werden.
- Bei großen Modellen kann der Speicherbedarf erheblich sein, was zu Performance-Problemen führen kann.
- Die Ladezeit kann je nach Netzwerkgeschwindigkeit und Modellgröße variieren.

**Hinweise**
- Um die Performance zu optimieren, sollten nur benötigte Modelle geladen werden.
- Sicherheitsaspekte sollten beachtet werden, insbesondere bei der Verwendung von externen Modellen.
- Regelmäßige Wartung des Caches ist empfohlen, um veraltete Modelle zu entfernen.
- Überwachung der Speichernutzung ist wichtig, um Engpässe zu vermeiden.

**Schnittstellen**
- **Input:** `model_name` (String) – Name des zu ladenden Modells.
- **Output:** Instanz von `SentenceTransformer` – Das geladene Modell für die Textverarbeitung.

### 4. Dataset Class
```python
class Dataset(Dataset):
    def __init__(self, X, y, w=None):
        if isinstance(X, torch.Tensor):
            self.X = X.float()
        else:
            self.X = torch.from_numpy(X.astype(np.float32))
        if isinstance(y, torch.Tensor):
            self.y = y.long()
        else:
            self.y = torch.from_numpy(y.astype(np.int64))
        if w is not None:
            if isinstance(w, torch.Tensor):
                self.w = w.float()
            else:
                self.w = torch.from_numpy(w.astype(np.float32))
        else:
            self.w = None

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        if self.w is not None:
            return self.X[idx], self.y[idx], self.w[idx]
        return self.X[idx], self.y[idx]
```
**Erklärung**
- Die `Dataset`-Klasse dient zur Erstellung eines benutzerdefinierten Datasets für maschinelles Lernen, das Daten in Form von Eingabematrizen `X`, Zielwerten `y` und optionalen Gewichten `w` lädt.
- Im Konstruktor (`__init__`) werden die Eingabedaten in Tensoren umgewandelt, um sicherzustellen, dass sie im richtigen Format für PyTorch vorliegen.
- Die Methoden `__len__` und `__getitem__` ermöglichen die Interaktion mit der Klasse, indem sie die Länge des Datasets zurückgeben und den Zugriff auf spezifische Datenelemente ermöglichen.

**Besonderheiten & Randfälle**
- Unterstützung für Eingaben sowohl als NumPy-Arrays als auch als PyTorch-Tensoren.
- Konvertierung von Datentypen (z.B. `float32` für `X` und `int64` für `y`).
- Optionale Gewichte `w`, die ebenfalls als Tensoren oder NumPy-Arrays übergeben werden können.
- Fehlerbehandlung bei ungültigen Datentypen ist nicht implementiert.
- Bei `None`-Werten für `w` wird eine alternative Rückgabe in `__getitem__` verwendet.
- Die Klasse erfordert, dass `X` und `y` die gleiche Länge haben.

**Hinweise**
- Achten Sie auf die Konsistenz der Datentypen, um Laufzeitfehler zu vermeiden.
- Die Verwendung von `torch.from_numpy` kann zu Speicherproblemen führen, wenn die NumPy-Arrays nicht im richtigen Format sind.
- Die Klasse könnte um Validierungslogik erweitert werden, um sicherzustellen, dass die Eingabedaten korrekt sind.
- Bei großen Datensätzen kann die Umwandlung in Tensoren speicherintensiv sein; eine Lazy-Loading-Strategie könnte in Betracht gezogen werden.

**Schnittstellen**
- Eingaben: `X` (Features), `y` (Labels), `w` (optionale Gewichte).
- Ausgaben: Zugriff auf Datenelemente über `__getitem__`, Rückgabe von Tupeln `(X[idx], y[idx], w[idx])` oder `(X[idx], y[idx])`.

### 5. Load Cached DataFrame Function
```python
@st.cache_data
def load_cached_dataframe(file_path: str, source: str = "local"):
    if source == "local":
        return pd.read_csv(file_path)
    else:
        return file_path
```
**Erklärung**
- Die Funktion `load_cached_dataframe` lädt ein DataFrame aus einer CSV-Datei und nutzt Caching, um die Leistung zu optimieren.
- Sie akzeptiert zwei Parameter: `file_path`, der den Pfad zur CSV-Datei angibt, und `source`, der standardmäßig auf "local" gesetzt ist.
- Wenn die Quelle "local" ist, wird die CSV-Datei mit `pd.read_csv` geladen; andernfalls wird der `file_path` direkt zurückgegeben.
- Das Caching ermöglicht eine schnellere Datenverarbeitung bei wiederholtem Zugriff auf dieselbe Datei.

**Besonderheiten & Randfälle**
- Funktioniert nur mit lokalen CSV-Dateien, wenn `source` auf "local" gesetzt ist.
- Bei ungültigem `file_path` kann ein Fehler beim Laden der Datei auftreten.
- Wenn `source` nicht "local" ist, wird kein DataFrame geladen, sondern nur der Pfad zurückgegeben.
- Caching kann bei Änderungen der CSV-Datei zu veralteten Daten führen, wenn nicht neu geladen wird.
- Die Funktion unterstützt keine anderen Dateiformate außer CSV.
- Es gibt keine Fehlerbehandlung für fehlgeschlagene Lesevorgänge.

**Hinweise**
- Caching verbessert die Leistung, sollte jedoch mit Bedacht verwendet werden, um veraltete Daten zu vermeiden.
- Bei großen CSV-Dateien kann der Speicherbedarf erheblich sein; daher sollte der verfügbare Speicherplatz berücksichtigt werden.
- Sicherheitsaspekte sollten beachtet werden, insbesondere bei der Verarbeitung von Daten aus unsicheren Quellen.
- Regelmäßige Wartung und Überprüfung der CSV-Dateien sind notwendig, um Datenintegrität sicherzustellen.

**Schnittstellen**
- **Input**: `file_path` (String), `source` (String, optional).
- **Output**: DataFrame (bei `source` = "local") oder String (bei anderen Quellen).

### 6. Streamlit UI Setup
```python
st.title("👟 Allgemeines Training")
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
st.subheader("➕ Neuen Kunden hinzufügen")
with st.form("add_row_form"):
    col1, col2 = st.columns(2)
    with col1:
        kundename = st.text_input("Kundename", placeholder="z.B. TechCorp GmbH")
        land = st.selectbox("Land", ["Deutschland", "USA", "Frankreich", "UK", "Spanien", "Italien", "Schweiz", "Österreich", "Niederlande", "Belgien"])
        zeit = st.text_input("Zeit", value=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    with col2:
        prioritaet = st.selectbox("Priorität", ["Hoch", "Mittel", "Niedrig"])
        description = st.text_area("Beschreibung", height=100, placeholder="Beschreibung des Anliegens...")
    submitted = st.form_submit_button("Kunden hinzufügen")
```
**Erklärung**
- Dieses Segment richtet die Benutzeroberfläche für die Streamlit-Anwendung ein, um neue Kunden hinzuzufügen.
- Es wird ein Titel und eine Unterüberschrift gesetzt, gefolgt von einem Formular, das Eingabefelder für Kundendaten bereitstellt.
- Die Eingabefelder umfassen den Kundennamen, das Land, die Zeit, die Priorität und eine Beschreibung.
- Das Formular ermöglicht es dem Benutzer, die eingegebenen Daten zu überprüfen und zu bestätigen, bevor sie gespeichert werden.

**Besonderheiten & Randfälle**
- Überprüfung, ob der `uploader_key` im `session_state` vorhanden ist, um den Zustand zwischen den Sitzungen zu speichern.
- Verwendung von `datetime.now()` zur automatischen Zeitstempelung, was zu inkonsistenten Zeitformaten führen kann, wenn nicht richtig behandelt.
- Platzhaltertexte in den Eingabefeldern bieten zusätzliche Hinweise zur erwarteten Eingabe.
- Die Auswahlmöglichkeiten für das Land und die Priorität sind festgelegt, was die Eingabe vereinfacht, aber auch die Flexibilität einschränkt.
- Das Formular wird nur abgesendet, wenn der Benutzer auf den Button klickt, was eine bewusste Entscheidung zur Dateneingabe erfordert.
- Mögliche Probleme bei der Validierung der Eingaben sind nicht behandelt.

**Hinweise**
- Achten Sie darauf, die Eingaben auf Validität zu überprüfen, um unerwartete Fehler zu vermeiden.
- Berücksichtigen Sie die Performance, wenn viele Benutzer gleichzeitig auf das Formular zugreifen.
- Sensible Daten sollten sicher gespeichert und verarbeitet werden, um Datenschutzrichtlinien einzuhalten.
- Regelmäßige Wartung des Codes und der Benutzeroberfläche ist erforderlich, um die Benutzerfreundlichkeit zu gewährleisten.

**Schnittstellen**
- Eingaben aus diesem Segment werden wahrscheinlich an eine Datenbank oder ein Backend-System zur Speicherung der Kundendaten weitergeleitet.
- Die `session_state`-Verwendung ermöglicht die Interaktion mit anderen Segmenten, die möglicherweise auf den `uploader_key` zugreifen.

### 7. Add Customer Logic
```python
if submitted:
    if kundename.strip() and description.strip():
        try:
            new_row = pd.DataFrame({
                'Kundename': [kundename],
                'Land': [land],
                'Zeit': [zeit],
                'Priorität': [prioritaet],
                'description': [description]
            })
            if 'df' in st.session_state:
                st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            else:
                st.session_state.df = new_row
            if os.path.exists(local_csv_path):
                try:
                    existing_df = pd.read_csv(local_csv_path)
                    updated_df = pd.concat([existing_df, new_row], ignore_index=True)
                    updated_df.to_csv(local_csv_path, index=False)
                    st.success("✅ Kunde erfolgreich hinzugefügt und gespeichert!")
                except Exception as e:
                    st.warning(f"⚠️ Kunde hinzugefügt, aber Speicherung fehlgeschlagen: {e}")
            else:
                new_row.to_csv(local_csv_path, index=False)
                st.success("✅ Kunde erfolgreich hinzugefügt!")
            st.rerun()
        except Exception as e:
            st.error(f"Fehler beim Hinzufügen: {e}")
    else:
        st.error("❌ Bitte gib Kundename und Beschreibung ein.")
```
**Erklärung**
- Dieses Code-Segment verarbeitet die Logik zum Hinzufügen eines neuen Kunden in ein DataFrame und speichert die Daten in einer CSV-Datei.
- Es prüft, ob die Eingabefelder für den Kundennamen und die Beschreibung ausgefüllt sind.
- Bei erfolgreicher Validierung wird ein neuer DataFrame erstellt und entweder zu einem bestehenden DataFrame in der Session oder als neuer DataFrame gespeichert.
- Der Code versucht, die Daten in einer CSV-Datei zu speichern und gibt entsprechende Erfolgsmeldungen oder Warnungen aus.

**Besonderheiten & Randfälle**
- Eingabefelder dürfen nicht leer sein; sonst wird eine Fehlermeldung ausgegeben.
- Bei der Speicherung wird geprüft, ob die CSV-Datei existiert; falls nicht, wird sie neu erstellt.
- Fehler beim Lesen oder Schreiben der CSV-Datei werden abgefangen und führen zu Warnmeldungen.
- Es wird eine Rückmeldung an den Benutzer gegeben, ob der Kunde erfolgreich hinzugefügt wurde.
- Bei mehrfachen Aufrufen wird das DataFrame in der Session aktualisiert.
- Bei einem Fehler während des Hinzufügens wird eine Fehlermeldung angezeigt.

**Hinweise**
- Achten Sie auf die Validierung der Eingabewerte, um unerwartete Fehler zu vermeiden.
- Die Verwendung von `pd.concat` kann bei sehr großen DataFrames die Performance beeinträchtigen.
- Stellen Sie sicher, dass der Pfad zur CSV-Datei korrekt ist, um Speicherfehler zu vermeiden.
- Berücksichtigen Sie Sicherheitsaspekte beim Umgang mit Benutzereingaben, um SQL-Injection oder ähnliche Angriffe zu verhindern.

**Schnittstellen**
- **Input:** `kundename`, `land`, `zeit`, `prioritaet`, `description` (Benutzereingaben).
- **Output:** Aktualisiertes DataFrame in `st.session_state`, CSV-Datei an `local_csv_path`.

### 8. Upload Table Logic
```python
with st.form("Tabelle anhängen"):
    uploaded_file = st.file_uploader("Hänge ein ganze Tabelle an die bestehende!", type=["csv"])
    submitted = st.form_submit_button("Tabelle anhängen")
    if submitted:
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                if 'df' in st.session_state and st.session_state.df is not None:
                    st.session_state.df = pd.concat([st.session_state.df, new_data], ignore_index=True)
                else:
                    st.session_state.df = new_data
            except Exception as e:
                st.error(f"Fehler beim Anhängen der Tabelle: {e}")
```
**Erklärung**
- Dieses Code-Segment ermöglicht das Hochladen einer CSV-Datei, die an ein bestehendes DataFrame angehängt wird.
- Es wird ein Formular erstellt, in dem der Benutzer eine Datei auswählen und das Hochladen initiieren kann.
- Nach dem Hochladen wird die Datei eingelesen und, falls ein DataFrame im `session_state` vorhanden ist, mit diesem kombiniert.
- Bei Fehlern während des Lesevorgangs wird eine Fehlermeldung angezeigt.

**Besonderheiten & Randfälle**
- Überprüfung, ob die hochgeladene Datei tatsächlich eine CSV-Datei ist.
- Handhabung des Falls, dass kein DataFrame im `session_state` existiert.
- Fehlerbehandlung für ungültige CSV-Dateien oder Lesefehler.
- Möglichkeit, leere oder nicht kompatible DataFrames zu verarbeiten.
- Sicherstellen, dass die Spalten der neuen Tabelle mit denen des bestehenden DataFrames übereinstimmen.
- Berücksichtigung von Duplikaten, falls diese in den neuen Daten vorhanden sind.

**Hinweise**
- Die Performance kann bei sehr großen CSV-Dateien beeinträchtigt werden; eine Vorverarbeitung könnte sinnvoll sein.
- Sicherheitsaspekte: Validierung der Dateiinhalte, um schädliche Daten zu vermeiden.
- Wartung: Regelmäßige Überprüfung der Datenintegrität nach dem Anhängen neuer Daten.
- Nutzung von `ignore_index=True` in `pd.concat`, um Indexkonflikte zu vermeiden.

**Schnittstellen**
- **Input:** CSV-Datei über den `file_uploader`.
- **Output:** Aktualisiertes DataFrame im `session_state`, das die neuen Daten enthält.

### 9. Load CSV Logic
```python
if 'uploaded_file' in st.session_state:
    del st.session_state.uploaded_file
st.session_state.uploader_key += 1
st.session_state.current_source = "local"
local_csv_path = st.text_input("Gib den Pfad zur CSV-Datei ein:", "beispiel_daten")
local_csv_path += ".csv"
if st.button("Laden"):
    try:
        df = load_cached_dataframe(local_csv_path, "local")
        st.session_state.df = df
        st.success(f"✅ Lokale CSV geladen: {len(df)} Kunden")
    except Exception as e:
        st.error(f"Fehler beim Laden der lokalen CSV: {e}")
```
**Erklärung**
- Dieses Code-Segment ermöglicht das Laden einer CSV-Datei aus dem lokalen Dateisystem in die Anwendung.
- Zunächst wird geprüft, ob eine vorherige Datei im Session-State vorhanden ist, die dann gelöscht wird.
- Der Benutzer gibt den Pfad zur CSV-Datei ein, und beim Klicken auf den "Laden"-Button wird die Datei geladen.
- Bei erfolgreichem Laden wird die Anzahl der geladenen Kunden angezeigt, andernfalls wird eine Fehlermeldung ausgegeben.

**Besonderheiten & Randfälle**
- Der Pfad zur CSV-Datei muss korrekt eingegeben werden, einschließlich der Dateiendung ".csv".
- Es wird keine Validierung des Dateiformats oder der Datenstruktur vor dem Laden durchgeführt.
- Bei einem Fehler während des Ladevorgangs wird eine generische Fehlermeldung angezeigt.
- Der Session-State wird aktualisiert, was zu unerwartetem Verhalten führen kann, wenn mehrere Ladevorgänge hintereinander durchgeführt werden.
- Es gibt keine Überprüfung, ob die Datei tatsächlich existiert, bevor der Ladevorgang initiiert wird.
- Der Benutzer muss sicherstellen, dass die Datei im richtigen Verzeichnis liegt.

**Hinweise**
- Um die Performance zu verbessern, sollte eine Caching-Strategie für häufig verwendete CSV-Dateien in Betracht gezogen werden.
- Sicherheitsaspekte wie die Validierung des Dateipfades sind wichtig, um Angriffe durch Pfadmanipulation zu vermeiden.
- Regelmäßige Wartung des Codes ist erforderlich, um sicherzustellen, dass die Fehlerbehandlung aktuell und informativ bleibt.
- Eine Benutzerführung zur Eingabe des Dateipfades könnte die Benutzerfreundlichkeit erhöhen.

**Schnittstellen**
- **Input**: Benutzer gibt den Pfad zur CSV-Datei über ein Textfeld ein.
- **Output**: Erfolgreiche Ladebestätigung oder Fehlermeldung wird im UI angezeigt; die DataFrame wird im Session-State gespeichert.

### 10. Temporary Info Message
```python
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
elapsed_time = time.time() - st.session_state.start_time
if elapsed_time < 5:
    st.info("ℹ️ Lade eine CSV-Datei über den Button oder Upload, um Daten anzuzeigen.")
```
**Erklärung**
- Dieses Code-Segment zeigt eine temporäre Informationsnachricht an, die für die ersten 5 Sekunden nach dem Laden der Seite sichtbar ist.
- Es wird überprüft, ob der Schlüssel `start_time` im `session_state` vorhanden ist; falls nicht, wird die aktuelle Zeit gespeichert.
- Die verstrichene Zeit wird berechnet, und die Nachricht wird nur angezeigt, solange diese Zeit weniger als 5 Sekunden beträgt.

**Besonderheiten & Randfälle**
- Die Nachricht wird nur einmal pro Sitzung angezeigt, da `start_time` nur einmal gesetzt wird.
- Bei einem Seitenneuladen wird die Nachricht erneut angezeigt, da `start_time` zurückgesetzt wird.
- Wenn die Seite länger als 5 Sekunden offen bleibt, wird die Nachricht nicht mehr angezeigt.
- Die Verwendung von `st.info` sorgt für eine visuelle Hervorhebung der Nachricht.
- Bei langsamen Verbindungen könnte die Nachricht möglicherweise nicht rechtzeitig angezeigt werden.
- Nutzer könnten die Nachricht als störend empfinden, wenn sie länger als 5 Sekunden benötigt wird.

**Hinweise**
- Die Performance ist in der Regel unkritisch, da die Berechnung der Zeit und die Anzeige der Nachricht minimalen Ressourcenverbrauch erfordert.
- Sicherheitsaspekte sind in diesem Segment nicht relevant, da keine Benutzereingaben verarbeitet werden.
- Wartung ist einfach, da der Code klar strukturiert ist und leicht angepasst werden kann.
- Es sollte darauf geachtet werden, dass die Nachricht für die Benutzer hilfreich und nicht irreführend ist.

**Schnittstellen**
- Input: Keine externen Eingaben, nur interne Zeitmessung.
- Output: Eine Informationsnachricht, die über die Streamlit-Funktion `st.info` angezeigt wird.

### 11. Display Data Logic
```python
if 'df' in st.session_state:
    df = st.session_state.df
    st.subheader("📋 Kunden-Daten")
    st.dataframe(df, use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gesamt Kunden", len(df))
    with col2:
        if 'Land' in df.columns:
            unique_countries = df['Land'].nunique()
            st.metric("Länder", unique_countries)
    with col3:
        if 'Priorität' in df.columns:
            high_priority = len(df[df['Priorität'] == 'Hoch'])
            st.metric("Hoch-Priorität", high_priority)
```
**Erklärung**
- Dieses Code-Segment zeigt Kundendaten und relevante Metriken an, die aus einem DataFrame stammen, der im Session-State gespeichert ist.
- Zunächst wird überprüft, ob der DataFrame (`df`) im Session-State vorhanden ist.
- Anschließend wird der DataFrame in einer Tabelle angezeigt, gefolgt von drei Metriken: der Gesamtzahl der Kunden, der Anzahl der einzigartigen Länder und der Anzahl der hochpriorisierten Kunden.
- Die Metriken werden in drei Spalten angeordnet, um eine übersichtliche Darstellung zu gewährleisten.

**Besonderheiten & Randfälle**
- Der Code prüft, ob der DataFrame im Session-State existiert, um Fehler zu vermeiden.
- Es wird sichergestellt, dass die Spalten 'Land' und 'Priorität' existieren, bevor auf sie zugegriffen wird.
- Bei einem leeren DataFrame wird die Gesamtzahl der Kunden als 0 angezeigt.
- Wenn keine Länder oder Prioritäten vorhanden sind, werden die entsprechenden Metriken nicht angezeigt.
- Der Code ist auf die Verwendung mit Streamlit optimiert, was eine spezifische Umgebung erfordert.
- Bei großen DataFrames kann die Darstellung der Daten in der Benutzeroberfläche langsam sein.

**Hinweise**
- Die Performance kann durch die Größe des DataFrames beeinträchtigt werden; eine Pagination könnte in Betracht gezogen werden.
- Sicherheitsaspekte sollten berücksichtigt werden, insbesondere beim Umgang mit sensiblen Kundendaten.
- Regelmäßige Wartung des Codes ist erforderlich, um sicherzustellen, dass die Spaltennamen im DataFrame aktuell sind.
- Eine Validierung der Daten vor der Anzeige könnte helfen, unerwartete Fehler zu vermeiden.

**Schnittstellen**
- **Input**: DataFrame (`df`) aus `st.session_state`.
- **Output**: Anzeige von Metriken und DataFrame in der Streamlit-Oberfläche.

### 12. Training Parameters Setup
```python
st.sidebar.subheader("⚙️ Trainingsparameter")
epochs = st.sidebar.slider("Epochen", 1, 100, 10)
lr = st.sidebar.select_slider("Learning Rate", options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-3)
bs = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128], value=128)
hidden_dim = st.sidebar.slider("Hidden Layer Größe", 50, 500, 100)
```
**Erklärung**
- Dieses Code-Segment erstellt eine Sidebar zur Konfiguration von Trainingsparametern für ein Machine Learning-Modell.
- Es ermöglicht dem Benutzer, die Anzahl der Epochen, die Lernrate, die Batch-Größe und die Größe der versteckten Schicht über interaktive Steuerelemente einzustellen.
- Die Parameter werden durch Slider und Auswahlfelder bereitgestellt, was eine benutzerfreundliche Anpassung ermöglicht.
- Standardwerte sind festgelegt, um einen sinnvollen Startpunkt für das Training zu bieten.

**Besonderheiten & Randfälle**
- Der Slider für Epochen hat einen minimalen Wert von 1, was sicherstellt, dass das Training nicht übersprungen wird.
- Die Lernrate ist auf vordefinierte Werte beschränkt, um extreme Werte zu vermeiden, die das Training destabilisieren könnten.
- Die Batch-Größe ist auf gängige Werte beschränkt, um die Effizienz des Trainings zu optimieren.
- Die Größe der versteckten Schicht kann zwischen 50 und 500 variieren, was Flexibilität bei der Modellarchitektur bietet.
- Bei extremen Werten könnte es zu Performance-Problemen kommen, insbesondere bei großen Batch-Größen.
- Die Benutzeroberfläche könnte überlastet sein, wenn zu viele Parameter gleichzeitig angepasst werden.

**Hinweise**
- Achten Sie darauf, dass die gewählten Parameter die Trainingszeit und -effizienz beeinflussen können.
- Eine zu hohe Lernrate kann zu instabilem Training führen, während eine zu niedrige Lernrate die Konvergenz verlangsamen kann.
- Die Sidebar sollte regelmäßig gewartet werden, um sicherzustellen, dass die Benutzeroberfläche intuitiv bleibt.
- Sicherheitsaspekte sollten berücksichtigt werden, um zu verhindern, dass Benutzer ungültige Parameter eingeben.

**Schnittstellen**
- **Input**: Benutzerinteraktionen über die Sidebar (Epochen, Lernrate, Batch-Größe, versteckte Schichtgröße).
- **Output**: Die konfigurierten Parameter werden an das Haupttraining-Skript übergeben, um das Modell entsprechend zu trainieren.

### 13. Data Preparation
```python
device = "mps"
backbone = load_backbone().to(device)
if st.toggle("Keywords anwenden"):
    medium = st.text_input("Keywords für 'medium' Priorität: ", "")
    high = st.text_input("Keywords für 'high' Priorität: ", "")
    high_pattern = '|'.join(high.split()) if high else ''
    med_pattern = '|'.join(medium.split()) if medium else ''
    text_lc = df['description'].str.lower()
    conds = [text_lc.str.contains(high_pattern, regex=True), text_lc.str.contains(med_pattern, regex=True)]
    df['sample_weight'] = np.select(conds, [1.5, 1.2], default=1.0)
    df['Priorität'] = np.select(conds, ['high', 'medium'], default=df['Priorität'])
else:
    df['sample_weight'] = 1.0
```
**Erklärung**
- Dieses Segment bereitet die Daten für das Training vor, indem es Schlüsselwörter anwendet, um die Priorität der Datenpunkte zu bestimmen.
- Es wird überprüft, ob die Anwendung von Schlüsselwörtern aktiviert ist. Wenn ja, werden die Eingaben für "medium" und "high" Priorität erfasst.
- Die Schlüsselwörter werden in reguläre Ausdrücke umgewandelt, um die entsprechenden Zeilen in der Beschreibung zu identifizieren.
- Basierend auf den gefundenen Schlüsselwörtern werden die Gewichtungen (`sample_weight`) und die Priorität (`Priorität`) der Datenpunkte angepasst.

**Besonderheiten & Randfälle**
- Wenn keine Schlüsselwörter eingegeben werden, bleibt die Priorität unverändert.
- Die Verwendung von regulären Ausdrücken kann zu unerwarteten Ergebnissen führen, wenn die Eingaben nicht korrekt formatiert sind.
- Bei großen Datensätzen kann die Verarbeitung der Textsuche zeitintensiv sein.
- Die Eingabe von Schlüsselwörtern ist optional; das Segment funktioniert auch ohne sie.
- Die Groß-/Kleinschreibung wird durch die Umwandlung in Kleinbuchstaben ignoriert.
- Es wird keine Validierung der eingegebenen Schlüsselwörter durchgeführt.

**Hinweise**
- Achten Sie auf die Performance bei der Verarbeitung großer DataFrames, insbesondere bei der Verwendung von `str.contains()`.
- Sicherheitsaspekte sollten berücksichtigt werden, um SQL-Injection oder andere Angriffe durch unsichere Eingaben zu vermeiden.
- Regelmäßige Wartung des Codes ist erforderlich, um sicherzustellen, dass die Regex-Muster aktuell und relevant bleiben.
- Dokumentation der Schlüsselwörter und ihrer Bedeutung kann die Wartung und Nutzung des Codes erleichtern.

**Schnittstellen**
- **Input:** `df['description']` (DataFrame mit Beschreibungen), `medium`, `high` (Benutzereingaben für Schlüsselwörter).
- **Output:** `df['sample_weight']` (angepasste Gewichtungen), `df['Priorität']` (aktualisierte Priorität der Datenpunkte).

### 14. Model Training Logic
```python
if st.button("🤖 KI-Modell trainieren"):
    ...
    for epoch in range(epochs):
        ...
        for (x_t,y_t,w_t),(x_p,y_p,w_p) in zip(t_dl, p_dl):
            ...
            loss.backward()
            ...
    st.success("Training abgeschlossen!")
```
**Erklärung**
- Dieses Code-Segment implementiert die Logik zum Trainieren eines KI-Modells, ausgelöst durch einen Button-Klick in einer Streamlit-Anwendung.
- Es initialisiert das Modell, den Optimierer und die Verlustfunktion, und bereitet die Datensätze für Produkte und Prioritäten vor.
- In einer Schleife über die Epochen wird das Modell trainiert, indem es die Eingabedaten verarbeitet, den Verlust berechnet und die Gewichte aktualisiert.
- Der Fortschritt wird visuell angezeigt, und nach Abschluss des Trainings wird eine Erfolgsmeldung ausgegeben.

**Besonderheiten & Randfälle**
- Das Training erfolgt in Batches, was die Speichereffizienz erhöht.
- Verlust wird für zwei unterschiedliche Datensätze (Produkte und Prioritäten) berechnet und kombiniert.
- Bei unzureichendem Speicher kann es zu einem Absturz kommen, wenn das Modell oder die Daten nicht auf das Gerät passen.
- Die Verwendung von `torch.optim.Adam` ermöglicht adaptives Lernen, was bei unterschiedlichen Lernraten vorteilhaft ist.
- Fortschrittsanzeige und Status-Updates sind in Echtzeit implementiert, was die Benutzererfahrung verbessert.
- Bei extremen Verlustwerten könnte das Training instabil werden.

**Hinweise**
- Achten Sie auf die Wahl der Hyperparameter (z.B. Lernrate, Batch-Größe), da diese die Trainingsqualität erheblich beeinflussen.
- Die Verwendung von `torch.no_grad()` könnte in der Validierungsphase sinnvoll sein, um den Speicherverbrauch zu reduzieren.
- Regelmäßige Speicherung des Modells während des Trainings kann bei unerwarteten Unterbrechungen hilfreich sein.
- Überwachen Sie den Verlust, um Überanpassung zu vermeiden; eventuell sollten Validierungsdaten integriert werden.

**Schnittstellen**
- **Input:** Button-Klick von Streamlit, Datensätze `data['product']` und `data['priority']`.
- **Output:** Fortschrittsanzeige, Status-Updates und eine Erfolgsmeldung nach Abschluss des Trainings.

### 15. Evaluation Logic
```python
model.eval()
with torch.no_grad():
    X_test_t = torch.as_tensor(data['product']['X_test'], dtype=torch.float32, device=device)
    X_test_p = torch.as_tensor(data['priority']['X_test'], dtype=torch.float32, device=device)
    pred_t = torch.argmax(model(X_test_t, 0), dim=1).cpu().numpy()
    pred_p = torch.argmax(model(X_test_p, 1), dim=1).cpu().numpy()
    y_test_product_np = data['product']['y_test'].numpy()
    y_test_priority_np = data['priority']['y_test'].numpy()
    st.subheader("🏁 Trainingsergebnisse")
    col1, col2 = st.columns(2)
    col1.metric("Product F1-Score", f"{f1_score(y_test_product_np, pred_t, average='weighted'):.4f}")
    col2.metric("Priority F1-Score", f"{f1_score(y_test_priority_np, pred_p, average='weighted'):.4f}")
```
**Erklärung**
- Dieses Code-Segment evaluiert ein trainiertes Modell, indem es Vorhersagen für Testdaten generiert und die F1-Scores für zwei verschiedene Klassifikationen (Produkt und Priorität) berechnet.
- Der Evaluationsprozess erfolgt im `eval()`-Modus, um sicherzustellen, dass das Modell keine Gradienten berechnet, was die Performance verbessert.
- Die Vorhersagen werden durch die Verwendung von `torch.argmax` ermittelt, um die Klassen mit der höchsten Wahrscheinlichkeit auszuwählen.
- Die Ergebnisse werden in einer Streamlit-Oberfläche angezeigt, wobei die F1-Scores für beide Klassifikationen in zwei Spalten dargestellt werden.

**Besonderheiten & Randfälle**
- Verwendung von `torch.no_grad()`, um den Speicherverbrauch zu reduzieren und die Berechnungszeit zu optimieren.
- F1-Score wird mit `average='weighted'` berechnet, was wichtig ist, wenn die Klassen unausgewogen sind.
- Vorhersagen werden auf die CPU übertragen, was bei der Verwendung von GPUs wichtig ist.
- Mögliche Fehler bei der Konvertierung von Tensoren in NumPy-Arrays, wenn die Dimensionen nicht übereinstimmen.
- Streamlit könnte bei großen Datenmengen Performance-Probleme aufweisen.
- Sicherstellen, dass die Testdaten im richtigen Format vorliegen, um Laufzeitfehler zu vermeiden.

**Hinweise**
- Überprüfen Sie die Konsistenz der Testdatenformate, um Fehler zu vermeiden.
- Berücksichtigen Sie die Performance bei der Verwendung von großen Datensätzen; eventuell Batch-Verarbeitung in Betracht ziehen.
- Achten Sie auf die Sicherheit der Daten, insbesondere bei sensiblen Informationen in den Testdaten.
- Regelmäßige Wartung des Modells und der Evaluationslogik ist notwendig, um die Genauigkeit über Zeit zu gewährleisten.

**Schnittstellen**
- **Input:** `data['product']['X_test']`, `data['priority']['X_test']`, `data['product']['y_test']`, `data['priority']['y_test']`
- **Output:** F1-Scores für Produkt und Priorität, angezeigt in der Streamlit-Oberfläche.

### 16. Classification Report Logic
```python
with st.expander("Classification Report ansehen"):
    st.subheader("Product Classification Report")
    report_p = classification_report(
        y_test_product_np, pred_t, 
        target_names=data['encoders']['product'].classes_, 
        labels=range(data['n_classes']['product']),
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report_p).transpose())
    st.subheader("Priority Classification Report")
    report_pr = classification_report(
        y_test_priority_np, pred_p, 
        target_names=data['encoders']['priority'].classes_, 
        labels=range(data['n_classes']['priority']),
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report_pr).transpose())
```
**Erklärung**
- Dieses Code-Segment generiert und zeigt Klassifikationsberichte für Produkt- und Prioritätsvorhersagen an.
- Es verwendet die Funktion `classification_report` aus der Bibliothek `sklearn`, um die Leistung der Modelle zu bewerten.
- Die Berichte beinhalten Metriken wie Präzision, Recall und F1-Score und werden in einem DataFrame für die Anzeige aufbereitet.
- Die Berichte werden in einem interaktiven Streamlit-Expander präsentiert, um die Benutzeroberfläche übersichtlich zu halten.

**Besonderheiten & Randfälle**
- Berichte werden nur angezeigt, wenn die Vorhersagen (`pred_t`, `pred_p`) und die Testdaten (`y_test_product_np`, `y_test_priority_np`) korrekt dimensioniert sind.
- Bei unzureichenden Daten (z.B. keine positiven Klassen) können die Metriken undefiniert sein.
- Die Verwendung von `output_dict=True` ermöglicht eine einfache Umwandlung in ein DataFrame, was die Flexibilität erhöht.
- Die Labels müssen mit den Klassen übereinstimmen, andernfalls kann es zu Fehlern kommen.
- Die Funktion kann bei sehr großen Datensätzen langsam sein, da sie alle Metriken berechnet.
- Die Ausgabe ist abhängig von der korrekten Konfiguration der Encoder in `data['encoders']`.

**Hinweise**
- Achten Sie auf die Performance, insbesondere bei großen Datensätzen, um lange Ladezeiten zu vermeiden.
- Sicherheitsaspekte sollten berücksichtigt werden, insbesondere beim Umgang mit Benutzereingaben und Daten.
- Regelmäßige Wartung der Encoder-Klassen ist erforderlich, um sicherzustellen, dass sie mit den aktuellen Daten übereinstimmen.
- Die Verwendung von `st.dataframe` ermöglicht eine interaktive Ansicht, die jedoch bei großen DataFrames die Benutzererfahrung beeinträchtigen kann.

**Schnittstellen**
- **Input**: 
  - `y_test_product_np`: Numpy-Array mit den tatsächlichen Produktlabels.
  - `pred_t`: Numpy-Array mit den vorhergesagten Produktlabels.
  - `y_test_priority_np`: Numpy-Array mit den tatsächlichen Prioritätslabels.
  - `pred_p`: Numpy-Array mit den vorhergesagten Prioritätslabels.
  - `data`: Dictionary mit Encoder-Informationen und Klassenzahlen.
  
- **Output**: 
  - Zwei DataFrames, die die Klassifikationsberichte für Produkt- und Prioritätsvorhersagen darstellen.

### 17. Download Functionality
```python
csv = df.to_csv(index=False)
st.download_button(
    label="📥 CSV herunterladen",
    data=csv,
    file_name=f"kunden_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)
```
**Erklärung**
- Diese Funktion ermöglicht es Benutzern, einen DataFrame als CSV-Datei herunterzuladen.
- Der DataFrame wird zunächst in eine CSV-Zeichenkette umgewandelt, wobei der Index ausgeschlossen wird.
- Anschließend wird ein Download-Button erstellt, der es dem Benutzer ermöglicht, die CSV-Datei mit einem zeitstempelbasierten Dateinamen herunterzuladen.
- Der MIME-Typ wird auf "text/csv" gesetzt, um den Browser über den Dateityp zu informieren.

**Besonderheiten & Randfälle**
- Der Dateiname enthält einen Zeitstempel, um Kollisionen bei gleichzeitigen Downloads zu vermeiden.
- Der Index des DataFrames wird nicht in die CSV-Datei aufgenommen, was die Lesbarkeit erhöht.
- Bei leeren DataFrames wird eine leere CSV-Datei generiert.
- Der Download-Button ist nur sichtbar, wenn der DataFrame Daten enthält.
- Bei großen DataFrames kann die Umwandlung in CSV viel Speicher benötigen.
- Der Benutzer muss über einen unterstützten Browser verfügen, um den Download erfolgreich abzuschließen.

**Hinweise**
- Die Performance kann bei sehr großen DataFrames beeinträchtigt werden; eine asynchrone Verarbeitung könnte in Betracht gezogen werden.
- Sicherheitsaspekte sollten beachtet werden, insbesondere bei sensiblen Daten im DataFrame.
- Regelmäßige Wartung des Codes ist erforderlich, um sicherzustellen, dass die Funktion mit zukünftigen Versionen von Streamlit kompatibel bleibt.
- Eine Validierung der Daten vor dem Download könnte implementiert werden, um sicherzustellen, dass nur gültige Daten exportiert werden.

**Schnittstellen**
- **Input:** DataFrame (`df`), der exportiert werden soll.
- **Output:** CSV-Datei, die vom Benutzer heruntergeladen wird.

---
*Generiert am:* 2025-09-28 16:53:37  
*Modell:* gpt-4o-mini • *Temp:* 0.2 • *Segmente:* 17