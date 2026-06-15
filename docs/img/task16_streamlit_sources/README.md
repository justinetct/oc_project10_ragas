# Captures — interface Streamlit (sources et limites)

Une capture par type de traitement, prise après la réponse de l'assistant.

| Fichier | Question testée | Ce que la capture montre |
|---|---|---|
| `01_sql_points.png` | Quel joueur a le plus de points ? | Badge « Traitement : SQL chiffres » + sources (Base SQLite NBA, SQL Tool) |
| `02_sql_3points.png` | Qui a le meilleur pourcentage à 3 points ? | Idem + ligne « Filtre 3P% : minimum 100 tentatives » |
| `03_rag_reddit.png` | Résume les discussions Reddit sur les favoris NBA. | Badge « RAG texte » + documents source (Reddit *.pdf + extrait) |
| `04_hybride.png` | Quel joueur a le plus de points et qu'est-ce que cela révèle de son rôle ? | Badge « Hybride (SQL + rédaction) » + « Chiffre vérifié : base SQLite NBA » |
| `05_hors_perimetre.png` | Quelle est la météo à Paris ? | Badge « Hors périmètre » + notice de recentrage |
| `06_info_manquante.png` | Quel joueur a le pire pourcentage à 3 points ? | Réponse « non pris en charge » + « aucun chiffre n'a été inventé » |
