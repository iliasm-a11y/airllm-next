# Reconstruction exacte des sources V4

Le reçu `receipt.json` rapporte une reconstruction dans un nouveau checkout
isolé, le 3 novembre 2026, à partir de llama.cpp
`3f8af15a76ac8675812e856da45454bafc90f12c` et des **16 patches dans l'ordre**
du manifeste de compilation V4. Les **13 fichiers natifs contrôlés** ont chacun
le SHA256 attendu. L'identité du checkout natif existant reste inchangée.

Cette sélection contient uniquement le reçu et le manifeste des empreintes.
Aucune source complète, bibliothèque, binaire ou poids n'est copié. Les
préfixes de chemins locaux sont remplacés par des alias ; `selection.json`
conserve tailles et SHA256 avant/après ce nettoyage.

La reconstruction ne compile rien, ne charge aucun modèle et n'émet aucune
commande GPU. Aucun nouveau binaire ni transfert de qualification numérique
n'est revendiqué : les preuves CPU8 et les bibliothèques mesurées existantes
conservent leur périmètre propre.
