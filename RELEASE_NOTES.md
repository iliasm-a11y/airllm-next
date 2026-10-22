# AirLLM Next 4.1.0rc1 — 22 octobre 2026

Version candidate pour les configurations validées, sans publication stable ni
fusion dans `main`. L'objectif d'inférence interactive de GLM-5.3/Qwen3.8 complets
sur 16 Go n'est pas certifié.

## Changements intégrés

- Backend Llama natif partageant les lectures sélectives, caches et sessions
  Next : GQA, RoPE standard, SiLU, embeddings liés ou distincts, KV borné.
- Projection du vocabulaire par tuiles adaptatives plafonnées par I/O et tenseur,
  libération avant la tuile suivante, préchargements périmés drainés. Défaut
  2 048 lignes retenu après comparaison 128/2 048 sur poids entraînés M5.
- Packs réversibles autonomes avec tous les tenseurs et assets tokenizer/config
  bornés et hachés ; conservation du mode overlay ; aucun code distant exécuté.
- Validation stricte des IDs/paramètres avant conversion, suffixe de logits
  correct entre chunks, bornes des espaces de travail GLM/Qwen/Llama.
- Correction de la porte officielle `swish` DeltaNet, distincte du sigmoid de
  l'attention complète ; backend Qwen toujours explicitement expérimental.
- Fermetures concurrentes sûres, commits de packs atomiques et reprise d'une
  écriture de manifeste sidecar interrompue.
- Prompt lookup en CLI et brouillons locaux gloutons GLM/Llama vérifiés par le
  cible, métriques logiques/encodées distinctes et budget cumulé des paires CLI.
- Durcissement du serveur local, configuration et diagnostic ; scripts macOS
  avec précision de validation explicite ; distributions wheel/sdist autonomes.

## Résultats et limites

Le checkpoint entraîné complet SmolLM2-135M-Instruct génère sur CPU et Metal.
Les séquences du protocole enregistré coïncident avec Transformers ; les logits
indépendants sont proches numériquement, sans revendication bit à bit générale.
La compression retrouve les bits originaux et réduit le stockage mesuré, mais
ralentit les lectures chaudes. Elle reste optionnelle. Le chemin GLM miniature
avant/après ne montre pas de gain général robuste.

Les détails, échecs conservés, décomptes uniques et mesures brutes sont reliés
dans [FINAL_VALIDATION.md](FINAL_VALIDATION.md). Le guide et la matrice de
compatibilité décrivent l'installation, le sous-ensemble API et les formats
non implémentés. NVFP4, MTP, sampling spéculatif, checkpoints géants, SSD externe
et longue validation thermique restent hors du périmètre démontré.
