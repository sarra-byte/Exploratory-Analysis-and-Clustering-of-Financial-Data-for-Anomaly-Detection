import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. CONFIGURATION ET STYLE (CSS PRO)
# ==========================================
st.set_page_config(
    page_title="Dashboard Détection Fraude", 
    layout="wide", 
    page_icon=None,
    initial_sidebar_state="expanded"
)

# Palette de couleurs "Corporate"
COULEURS = {
    "Normal": "#2980b9",   # Bleu
    "Anomalie": "#c0392b", # Rouge
    "Fraude": "#c0392b"
}

# Injection de CSS pour un look Dashboard épuré
st.markdown("""
<style>
    /* Fond global */
    .reportview-container { background: #f4f6f9; }
    
    /* Titres */
    h1, h2, h3 { font-family: 'Helvetica', sans-serif; color: #2c3e50; }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        padding-bottom: 10px;
        border-bottom: 2px solid #bdc3c7;
        margin-bottom: 25px;
    }
    
    /* Cartes d'information (Widgets) */
    .info-box {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #2980b9;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    
    /* Box d'analyse (Remarques résultats) */
    .analysis-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #7f8c8d;
        font-style: italic;
        font-size: 0.95rem;
        color: #444;
        margin-top: 10px;
    }

    /* Légende personnalisée */
    .legend-box {
        background-color: #fff;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }
    
    /* KPIs */
    .kpi-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        border: 1px solid #ecf0f1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .kpi-title { font-size: 0.85rem; color: #7f8c8d; text-transform: uppercase; font-weight: 600; }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #2c3e50; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CHARGEMENT DES DONNÉES
# ==========================================
@st.cache_data
def load_data():
    if not os.path.exists("results/sampled_data.csv"):
        return None, None, None, None
        
    X_pca = np.load("results/X_pca.npy")
    clusters = np.load("results/clusters.npy")
    labels = np.load("results/labels.npy")
    amounts = np.load("results/amounts.npy")
    raw_data = pd.read_csv("results/sampled_data.csv")
    with open("results/report_dict.pkl", "rb") as f:
        report_dict = pickle.load(f)
    summary = pd.read_csv("results/summary.csv")
    
    df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df["Cluster ID"] = clusters
    df["Résultat Modèle"] = df["Cluster ID"].apply(lambda x: "Anomalie" if x == -1 else "Normal")
    df["Montant"] = amounts
    df["Vérité Terrain"] = ["Fraude" if x == 1 else "Normal" for x in labels]
    
    return df, raw_data, report_dict, summary

df, raw_data, report_dict, summary = load_data()

if df is None:
    st.error("ERREUR : Fichiers de résultats introuvables. Lancez d'abord analyse.py.")
    st.stop()

# ==========================================
# 3. SIDEBAR (NAVIGATION) 
# ==========================================
with st.sidebar:
    st.title("Projet Data Mining")
    st.subheader("Détection Fraude Bancaire")
    st.markdown("---")
    
    choix = st.radio("Navigation :", [
        "1. Données & Prétraitement",
        "2. Analyse Exploratoire",
        "3. Clustering (DBSCAN)",
        "4. Interprétation",
        "5. Validation & KPIs"
    ])
    
    st.markdown("---")
    st.caption("Module Analyse de Données | Master 1")

# ==========================================
# PAGE 1 : DONNÉES & PRÉTRAITEMENT
# ==========================================
if choix == "1. Données & Prétraitement":
    st.markdown('<div class="main-title">Collecte et Préparation des Données</div>', unsafe_allow_html=True)
    
    st.subheader("Aperçu de l'échantillon de données")
    st.dataframe(raw_data.head().style.background_gradient(cmap="Blues", subset=["Amount"]))

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Impact de la Normalisation (StandardScaler)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box" style="border-left-color: #c0392b;">
        <b>Avant Normalisation</b><br>
        Les montants ont une variance trop élevée par rapport aux variables V1-V28 (Moyenne : {:.2f} €).
        Ceci risque de biaiser le calcul de distance euclidienne.
        </div>
        """.format(raw_data['Amount'].mean()), unsafe_allow_html=True)
        st.metric(label="Moyenne Montant (€)", value=f"{raw_data['Amount'].mean():.2f} €")
        st.metric(label="Écart-type Montant", value=f"{raw_data['Amount'].std():.2f}")
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <b>Après Normalisation</b><br>
        Toutes les variables sont centrées réduites (Moyenne=0, Écart-type=1). 
        Le modèle DBSCAN accordera désormais une importance égale au Montant et aux variables V1-V28.
        </div>
        """, unsafe_allow_html=True)
        st.success("Statut : Données normalisées et prêtes pour l'analyse.")

# ==========================================
# PAGE 2 : ANALYSE EXPLORATOIRE
# ==========================================
elif choix == "2. Analyse Exploratoire":
    st.markdown('<div class="main-title">Analyse Exploratoire (EDA)</div>', unsafe_allow_html=True)

    # Légende globale pour cette page
    st.markdown(f"""
    <div class="legend-box">
    <b>Légende Graphiques :</b> 
    <span style='color:{COULEURS['Normal']}; font-size:1.2em;'>■</span> Transactions Normales &nbsp;&nbsp;&nbsp;
    <span style='color:{COULEURS['Fraude']}; font-size:1.2em;'>■</span> Fraudes (Anomalies)
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Répartition des Classes")
        fig_pie = px.pie(
            df, 
            names="Vérité Terrain", 
            title=None,
            color="Vérité Terrain",
            color_discrete_map=COULEURS,
            hole=0.5
        )
        fig_pie.update_layout(template="plotly_white", margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("""
        <div class="analysis-box">
        <b>Observation :</b> Le déséquilibre extrême (Fraude < 1%) rend inefficaces les méthodes basées sur la simple classification. 
        DBSCAN est pertinent ici car il détecte la "rareté" sans avoir besoin d'un dataset équilibré.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Distribution des Montants (< 500€)")
        fig_hist = px.histogram(
            df[df["Montant"] < 500], 
            x="Montant", 
            color="Vérité Terrain",
            title=None,
            color_discrete_map=COULEURS,
            nbins=40,
            opacity=0.7,
            barmode="overlay"
        )
        fig_hist.update_layout(template="plotly_white", xaxis_title="Montant (€)", margin=dict(t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("""
        <div class="analysis-box">
        <b>Analyse :</b> Contrairement à l'intuition, les fraudes ne sont pas toujours des montants élevés. 
        On observe une concentration de fraudes sur les petits montants (stratégie de "test" des cartes bancaires).
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 3 : CLUSTERING DBSCAN
# ==========================================
elif choix == "3. Clustering (DBSCAN)":
    st.markdown('<div class="main-title">Résultats du Clustering DBSCAN</div>', unsafe_allow_html=True)

    col_viz, col_legend = st.columns([3, 1])

    with col_legend:
        st.markdown("#### Légende")
        st.markdown(f"""
        <div class="kpi-card" style="text-align:left; padding:15px;">
        <div><span style='color:{COULEURS['Normal']}; font-size:1.5em;'>■</span> <b>Normal</b><br><small>Transactions regroupées en clusters denses (Cluster 0)</small></div>
        <hr>
        <div><span style='color:{COULEURS['Anomalie']}; font-size:1.5em;'>■</span> <b>Anomalie</b><br><small>Points isolés rejetés par DBSCAN (Cluster -1)</small></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Métriques")
        st.metric("Total Transactions", len(df))
        st.metric("Anomalies Détectées", len(df[df['Résultat Modèle']=='Anomalie']))

    with col_viz:
        st.markdown("#### Projection PCA (2 Dimensions)")
        fig_pca = px.scatter(
            df, 
            x="PC1", 
            y="PC2", 
            color="Résultat Modèle",
            title=None,
            color_discrete_map=COULEURS,
            hover_data=["Montant", "Vérité Terrain"],
            opacity=0.8,
            height=500
        )
        fig_pca.update_traces(marker=dict(size=6))
        fig_pca.update_layout(template="plotly_white", margin=dict(t=10))
        st.plotly_chart(fig_pca, use_container_width=True)
        
        st.markdown("""
        <div class="analysis-box">
        <b>Interprétation spatiale :</b> La visualisation PCA confirme que les transactions normales (Bleu) forment un noyau dense et compact. 
        Les anomalies (Rouge) sont dispersées en périphérie : ce sont des points pour lesquels DBSCAN n'a pas trouvé suffisamment de voisins proches.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 4 : INTERPRÉTATION
# ==========================================
elif choix == "4. Interprétation":
    st.markdown('<div class="main-title">Analyse des Déviations</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Méthodologie :</b> Ce graphique identifie les variables qui rendent une transaction "suspecte". 
    Il montre l'écart moyen entre une transaction normale et une anomalie détectée.
    </div>
    """, unsafe_allow_html=True)

    # Calcul des différences
    df_temp = raw_data.copy()
    df_temp['Type'] = df['Résultat Modèle']
    numeric_cols = df_temp.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "Class"] 
    
    moyennes = df_temp.groupby('Type')[numeric_cols].mean().transpose()
    moyennes['Différence Absolue'] = abs(moyennes['Anomalie'] - moyennes['Normal'])
    top_diff = moyennes.sort_values(by='Différence Absolue', ascending=False).head(7)
    
    # Récupération du nom de la variable la plus influente
    top_var = top_diff.index[0]

    fig_bar = px.bar(
        top_diff, 
        y='Différence Absolue', 
        x=top_diff.index,
        title="Variables les plus discriminantes (Top 7)",
        color='Différence Absolue',
        color_continuous_scale='Reds',
        text_auto='.2f'
    )
    fig_bar.update_layout(template="plotly_white", xaxis_title="Variables", yaxis_title="Écart Moyen")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(f"""
    <div class="analysis-box">
    <b>Résultat Clé :</b> La variable <b>{top_var}</b> présente la divergence la plus forte. 
    Cela signifie que les fraudes détectées par ce modèle se caractérisent principalement par des valeurs anormales sur cet axe spécifique, 
    plus que sur le montant de la transaction lui-même.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 5 : EVALUATION
# ==========================================
elif choix == "5. Validation & KPIs":
    st.markdown('<div class="main-title">Performance du Modèle</div>', unsafe_allow_html=True)

    # KPIs
    metrics = summary.set_index("Métrique")["Valeur"]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-title">Fraudes Réelles</div><div class="kpi-value">{int(metrics["Fraudes réelles"])}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-title">Anomalies Détectées</div><div class="kpi-value" style="color:#c0392b">{int(metrics["Anomalies détectées"])}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-title">Rappel (Recall)</div><div class="kpi-value">{metrics["Rappel (fraude)"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-title">Silhouette Score</div><div class="kpi-value">{metrics["Silhouette Score"]:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    col_g, col_d = st.columns([1, 1])
    
    with col_g:
        st.subheader("Matrice de Confusion")
        y_true = df["Vérité Terrain"]
        y_pred = df["Résultat Modèle"].apply(lambda x: "Fraude" if x == "Anomalie" else "Normal")
        cm = confusion_matrix(y_true, y_pred, labels=["Normal", "Fraude"])
        
        fig_cm = px.imshow(
            cm, 
            x=["Prédit Normal", "Prédit Fraude"], 
            y=["Vrai Normal", "Vrai Fraude"], 
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(template="plotly_white", title=None)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown("""
        <div class="analysis-box">
        <b>Lecture :</b> 
        <ul>
            <li><b>Diagonale (Foncé) :</b> Prédictions correctes.</li>
            <li><b>Bas-Gauche (Faux Négatifs) :</b> Fraudes manquées par le système (Risque financier).</li>
            <li><b>Haut-Droite (Faux Positifs) :</b> Clients normaux bloqués par erreur (Expérience client).</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_d:
        st.subheader("Métriques Détaillées (Classe Fraude)")
        report = report_dict['1']
        metrics_df = pd.DataFrame({
            "Métrique": ["Précision", "Rappel", "F1-Score"],
            "Score": [report['precision'], report['recall'], report['f1-score']]
        })
        
        fig_metrics = px.bar(
            metrics_df, 
            x="Métrique", 
            y="Score", 
            text_auto='.2%',
            title=None,
            color="Score",
            color_continuous_scale="Teal",
            range_y=[0, 1]
        )
        fig_metrics.update_layout(template="plotly_white")
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        st.markdown(f"""
        <div class="analysis-box">
        <b>Conclusion Performance :</b> 
        Le rappel de <b>{report['recall']:.1%}</b> indique que le modèle a capturé {report['recall']:.1%} de l'ensemble des fraudes.
        C'est l'indicateur prioritaire dans un contexte de sécurité bancaire.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("Dashboard technique - Détection DBSCAN")