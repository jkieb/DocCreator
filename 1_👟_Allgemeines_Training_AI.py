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

# Klasse für Ai-Modell
class Model(nn.Module):
    def __init__(self, backbone: SentenceTransformer, hidden_dim: int, n_products: int, n_prios: int):
        super().__init__()
        self.backbone   = backbone
        emb_dim         = backbone.get_sentence_embedding_dimension()
        self.shared     = nn.Linear(emb_dim, hidden_dim)
        self.product_head = nn.Linear(hidden_dim, n_products)
        self.prio_head  = nn.Linear(hidden_dim, n_prios)

    def forward(self, features, task_id: int):
        # 1) If user already passed embeddings tensor, use it directly:
        if isinstance(features, torch.Tensor):
            embeddings = features

        # 2) If they passed raw text (list of strings or single string), call .encode():
        elif isinstance(features, (list, str)):
            embeddings = self.backbone.encode(features, convert_to_tensor=True)

        # 3) Otherwise assume it’s a tokenized dict and run through forward():
        else:
            output = self.backbone(features)
            if isinstance(output, dict):
                embeddings = output.get(
                    'sentence_embedding',
                    next(iter(output.values()))
                )
            else:
                embeddings = output

        # shared hidden layer + task head
        h = torch.relu(self.shared(embeddings))
        return self.product_head(h) if task_id == 0 else self.prio_head(h)

@st.cache_resource(show_spinner="Lade Basis-Modell...")
def load_backbone(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """Lädt das SentenceTransformer-Modell und cached es."""
    return SentenceTransformer(model_name)

# Dataset Klasse für den Dataloader
class Dataset(Dataset):
    def __init__(self, X, y, w=None):
        # Handle both numpy arrays and tensors
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




# Cache-Funktion für DataFrame
@st.cache_data
def load_cached_dataframe(file_path: str, source: str = "local"):
    """
    Lädt DataFrame aus Cache oder erstellt neuen Cache-Eintrag
    """
    if source == "local":
        return pd.read_csv(file_path)
    else:
        # Für uploaded files wird der Inhalt direkt übergeben
        return file_path

st.title("👟 Allgemeines Training")

# Session State für Uploader Key
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Neuen Datensatz hinzufügen.....
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
    
    if submitted:
        if kundename.strip() and description.strip():  # Prüfe ob Pflichtfelder ausgefüllt sind
            try:
                # Neue Zeile zum DataFrame hinzufügen
                new_row = pd.DataFrame({
                    'Kundename': [kundename],
                    'Land': [land],
                    'Zeit': [zeit],
                    'Priorität': [prioritaet],
                    'description': [description]
                })
                
                # DataFrame aktualisieren
                if 'df' in st.session_state:
                    st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
                else:
                    st.session_state.df = new_row
                
                # In lokale CSV speichern
                if os.path.exists(local_csv_path):
                    try:
                        # Bestehende Daten laden und neue Zeile hinzufügen
                        existing_df = pd.read_csv(local_csv_path)
                        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
                        updated_df.to_csv(local_csv_path, index=False)
                        st.success("✅ Kunde erfolgreich hinzugefügt und gespeichert!")
                    except Exception as e:
                        st.warning(f"⚠️ Kunde hinzugefügt, aber Speicherung fehlgeschlagen: {e}")
                else:
                    # Neue CSV-Datei erstellen
                    new_row.to_csv(local_csv_path, index=False)
                    st.success("✅ Kunde erfolgreich hinzugefügt!")
                
                st.rerun()
            except Exception as e:
                st.error(f"Fehler beim Hinzufügen: {e}")
        else:
            st.error("❌ Bitte gib Kundename und Beschreibung ein.")

# ...oder Tabelle hinzufügen
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


# Tabelle laden
st.subheader("📁 CSV-Datei (im selben Folder) laden (alte Tabelle löschen)")
# Uploaded file aus Session State entfernen und Key ändern
if 'uploaded_file' in st.session_state:
    del st.session_state.uploaded_file
st.session_state.uploader_key += 1  # Force uploader reset
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


# Temporäre Info-Nachricht für 5 Sekunden
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

# Zeige Info nur für die ersten 5 Sekunden
elapsed_time = time.time() - st.session_state.start_time
if elapsed_time < 5:
    st.info("ℹ️ Lade eine CSV-Datei über den Button oder Upload, um Daten anzuzeigen.")
# Daten anzeigen wenn vorhanden
if 'df' in st.session_state:
    df = st.session_state.df
    
    st.subheader("📋 Kunden-Daten")
    st.dataframe(df, use_container_width=True)
        
    # Übersicht
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
    
    base = 1.0
    w_med = 1.0
    w_high = 2.0 

    prio_map = {'low':0,'medium':1,'high':2,'critical':3}

    st.sidebar.subheader("⚙️ Trainingsparameter")
    epochs = st.sidebar.slider("Epochen", 1, 100, 10)
    lr = st.sidebar.select_slider("Learning Rate", options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-3)
    bs = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128], value=128)
    hidden_dim = st.sidebar.slider("Hidden Layer Größe", 50, 500, 100)

    # --- Data Preparation ---
    # This section runs regardless of the button press to ensure 'data' is always available.
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


    with torch.no_grad():
        descriptions = df['description'].tolist()
        emb = backbone.encode(descriptions, convert_to_tensor=True)
        w = df['sample_weight'].values


        product_encoder = LabelEncoder()
        y_t_encoded = product_encoder.fit_transform(df['product'].values)
        y_t_tensor = torch.tensor(y_t_encoded, dtype=torch.long)

        prio_encoder = LabelEncoder()
        y_p_encoded = prio_encoder.fit_transform(df['Priorität'].values)
        y_p_tensor = torch.tensor(y_p_encoded, dtype=torch.long)

        X_train, X_test, t_train, t_test, p_train, p_test, w_train, w_test = \
            train_test_split(emb, y_t_tensor, y_p_tensor, w, test_size=0.2, random_state=42)

        data = {
            'product':  {'X': X_train, 'y': t_train, 'X_test': X_test, 'y_test': t_test, 'w': w_train, 'w_test': w_test},
            'priority': {'X': X_train, 'y': p_train, 'X_test': X_test, 'y_test': p_test, 'w': w_train, 'w_test': w_test},
            'encoders': {'product': product_encoder, 'priority': prio_encoder},
            'n_classes': {'product': len(product_encoder.classes_), 'priority': len(prio_encoder.classes_)}
        }

    if st.button("🤖 KI-Modell trainieren"):
        st.write("Training wird gestartet...")

        model = Model(
            backbone, hidden_dim=hidden_dim,
            n_products=data['n_classes']['product'],
            n_prios=data['n_classes']['priority']
        ).to(device)

        st.toast(f"Modell wird trainiert.....")

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        
        t_ds = Dataset(data['product']['X'], data['product']['y'], data['product']['w'])
        p_ds = Dataset(data['priority']['X'], data['priority']['y'], data['priority']['w'])
        t_dl = DataLoader(t_ds, batch_size=bs, shuffle=True)
        p_dl = DataLoader(p_ds, batch_size=bs, shuffle=True)

        progress_bar = st.progress(0)
        status_text = st.empty()
        chart = st.line_chart()

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for (x_t,y_t,w_t),(x_p,y_p,w_p) in zip(t_dl, p_dl):
                x_t,y_t,w_t = x_t.to(device), y_t.to(device), w_t.to(device)
                x_p,y_p,w_p = x_p.to(device), y_p.to(device), w_p.to(device)
                l_t = (loss_fn(model(x_t,0), y_t) * w_t).mean()
                l_p = (loss_fn(model(x_p,1), y_p) * w_p).mean()
                loss = l_t + l_p
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(t_dl)
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            chart.add_rows([avg_loss])

        st.success("Training abgeschlossen!")

        model.eval()
        with torch.no_grad():
            # Uniforme Konvertierung, egal ob NumPy-Array oder Tensor
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


    # Download-Funktion
    st.subheader("💾 Daten exportieren")
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 CSV herunterladen",
        data=csv,
        file_name=f"kunden_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
