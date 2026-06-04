# Audit initial du prototype RAG

**Date :** 2026-06-04

## Structure observée

```
.
├── MistralChat.py              # Application Streamlit (interface chat RAG)
├── indexer.py                  # Script d'indexation des documents
├── requirements.txt            # Dépendances Python (pip)
├── inputs/                     # Documents sources
│   ├── Reddit 1.pdf
│   ├── Reddit 2.pdf
│   ├── Reddit 3.pdf
│   ├── Reddit 4.pdf
│   └── regular NBA.xlsx
├── utils/
│   ├── config.py               # Configuration (clé API, paramètres)
│   ├── data_loader.py          # Chargement et parsing des documents (PDF, DOCX, CSV, Excel, OCR)
│   └── vector_store.py         # Gestion de l'index FAISS et recherche sémantique
├── vector_db/                  # Index FAISS généré localement par indexer.py, non versionné
│   ├── faiss_index.idx
│   └── document_chunks.pkl
└── README.md                   # Documentation du prototype
```

## Commandes exécutées

### 1. Indexation des documents

```bash
python indexer.py
```

**Résultat :** Succès. L'OCR (EasyOCR) a été utilisé pour les 4 PDFs Reddit (captures d'écran, peu de texte extractible directement). Le fichier Excel a aussi été traité.

- 5 documents chargés et parsés (4 PDF + 1 Excel)
- 302 chunks créés
- Index FAISS généré dans `vector_db/`

**Observation :** L'indexation est lente (~3 minutes) à cause de l'OCR sur les PDFs. Le warning `pin_memory` de PyTorch/MPS est cosmétique et sans impact.

### 2. Lancement de l'application Streamlit

```bash
streamlit run MistralChat.py
```

**Résultat :** Succès. L'application démarre sur http://localhost:8501 et répond correctement (HTTP 200).

## Premières limites observées

### Architecture et code

1. **Version du SDK Mistral obsolète** : Le code utilise `mistralai==0.4.2` avec `from mistralai.client import MistralClient` (ancienne API). Les versions récentes du SDK Mistral (1.x) ont une API différente (`from mistralai import Mistral`). Cela fonctionne pour l'instant mais sera un problème de maintenance.

2. **Modules mentionnés dans le README mais absents** : `utils/database.py` et `utils/query_classifier.py` sont documentés dans le README mais n'existent pas dans le code. Le README décrit un projet plus complet que le prototype actuel.

> Le notebook `notebooks/audit.ipynb` ne fait **pas** partie des sources de l'énoncé.
>
> **Mise à jour (migration) :** le projet et le notebook utilisent désormais **Poetry** pour gérer les dépendances et la **dernière version du SDK `mistralai` (1.x)** à la place de l'ancien `0.4.2`. Le notebook s'exécute via le kernel Jupyter de l'environnement Poetry (et non plus l'ancien `.venv` pip).

### Données et qualité RAG

1. **Documents PDF = impressions d'écran Reddit** : Le contenu est extrait par OCR, ce qui introduit du bruit (erreurs de reconnaissance, mise en page Reddit mal parsée). La qualité des chunks est probablement médiocre.
2. **Fichier Excel converti en texte brut** : `regular NBA.xlsx` est transformé en texte via `pandas.to_string()` (cf. `utils/data_loader.py`). La structure du tableau est perdue et le découpage en chunks sépare les en-têtes de leurs valeurs. Résultat : un chiffre se retrouve isolé de son contexte, et le RAG ne peut ni agréger ni filtrer → besoin d'un Tool SQL.
3. **Réponses non ancrées dans les sources (*faithfulness*)** : le *system prompt* de `MistralChat.py` est permissif (« animer le débat ») et n'oblige **pas** le LLM à se limiter au contexte récupéré. 

    -> Résultat : sur les questions chiffrées (ex. « meilleur 3P% », « rebonds domicile vs extérieur »), le LLM produit des réponses confiantes **à partir de ses connaissances d'entraînement**, alors que ces chiffres ne sont **pas présents dans les données**. 

4. **Le RAG reformule, il ne calcule pas** : même lorsque le modèle cite une source existante, la réponse peut rester incorrecte. Par exemple, il répond Shai Gilgeous-Alexander pour le meilleur 3P%, alors que l'extrait affiché contient déjà Nikola Jokić à 41,7%. Le système ne calcule pas réellement le maximum de la colonne `3P%` : il reformule un chunk récupéré. Cela confirme le besoin d'un accès structuré aux données Excel via SQL.
5. **Pas de garde-fou sur les questions hors-sujet** : le prompt permissif (« animer le débat ») n'interdit pas les sujets hors NBA. Testé avec « Quelle est la recette de la ratatouille ? », le LLM **refuse parfois, mais donne souvent la recette** (comportement non déterministe, même à température 0.1). Le test a été reproduit avec l'ancien SDK (`0.4.2`) et le nouveau (`1.x`) : résultat identique → ce comportement vient du **prompt et du modèle**, pas de la migration. Un garde-fou (consigne de refus explicite, ou classification de la requête) serait nécessaire.
