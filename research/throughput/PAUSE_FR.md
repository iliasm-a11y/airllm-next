# Pause demandée — AirLLM Next Throughput

Travail arrêté le 24 octobre 2026 (heure de Paris), à la demande de l'utilisateur. Aucune nouvelle expérience à lancer avant sa demande de reprise. Aucun rappel ni réveil automatique programmé.

## État sauvegardé

- Branche locale `research/throughput-2026-10-24`, issue de la tête de la PR #5, `76fd683c37534807e21b5fda07a3d717b163215d`.
- Historique des PR #1 à #5 inspecté ; métadonnées enregistrées dans `research/throughput/pr-history.json`.
- Deux dernières inférences sans swap relues : preuves préexistantes `outputs/qwen36-20261023T221235Z` dans le répertoire principal de cette conversation.
- Profil natif réel achevé : `evidence-throughput/profile-baseline.json`. Principaux coûts exclusifs observés : évaluations/attentes MLX, lectures pread, gestion cache/buffers. Temps GPU isolé non disponible via l'API Python MLX installée ; les attentes synchrones ne sont pas un temps GPU pur.
- Prototype opt-in de regroupement des experts déjà routés : kernel Metal groupé et `GGUFWeights.linear_experts`, `NativeGGUFQwenModel(..., expert_batch=True)`. Réduction SIMD conservée par matrice et ordre d'addition des experts inchangé. Optimisation limitée au pas mono-token, repli vers le chemin existant pour le préremplissage multi-token et les poids flottants.
- Tests ciblés : 28 réussis (`test_gguf_throughput.py`, `test_gguf_metal.py`, `test_gguf_qwen.py`). La suite complète des 1 166 tests n'a pas encore été rejouée.
- Deux captures réelles de parité se sont achevées pendant la demande de pause : `parity-baseline` et `parity-expert-batch`. Aucun processus expérimental ne restait actif lorsque l'arrêt a été vérifié. Elles restent à analyser ; aucun gain confirmé annoncé.
- Aucun push, aucune PR créée, aucune fusion. Branche principale et preuves antérieures préservées.

## Reprise

1. Vérifier l'état Git, les ressources et les dernières sources. Lire cette note et `evidence-throughput/user-pause.json`.
2. Analyser les captures pilotes : logits complets, routes et états de génération ; ne pas assimiler leur débit à un benchmark apparié final.
3. Corriger si nécessaire le prototype et ajouter les contrôles de fidélité/sécurité manquants.
4. Effectuer les benchmarks de génération jusqu'à EOS, plusieurs longueurs de prompts, répétitions appariées et ordre mélangé. Conserver tous les contrôles mémoire inchangés, arrêter sur dépassement et ne pas relancer automatiquement un essai dangereux.
5. Tester d'autres optimisations seulement si les profils les justifient. Marquer précisément ce qui est mesuré et ce qui est non exploré.
6. Rejouer la suite complète et vérifier la CLI, l'instrumentation et les bornes mémoire du nouveau chemin.
7. Produire le rapport scientifique français et publier les améliorations utiles dans une PR brouillon dépendante de #5, autorisée par la mission utilisateur. Ne pas fusionner main.

## Limites connues du travail en cours

L'instrumentation est un outil de recherche et reste à compléter/tester. Les temps inclusifs ne doivent pas être additionnés ; la voie groupée n'est pas encore profilée par tous les wrappers. Les captures binaires locales ne doivent pas être publiées avec des poids. Pas de nouveau modèle, pas de SSD externe, pas de campagne thermique prolongée.
