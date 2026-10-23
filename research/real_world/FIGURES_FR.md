# Figures optionnelles — état et reproduction

Le script `benchmarks/plot_real_world.py` est prêt. **Aucune figure n’a été créée** : Matplotlib est absent des runtimes vérifiés en lecture seule, y compris le runtime fourni par Codex, Python Homebrew 3.12 et Python système 3.14. Aucun paquet n’a été installé et l’environnement de validation n’a pas été modifié. Les tableaux du rapport restent les résultats disponibles.

Le mode par défaut n’importe ni Matplotlib, ni NumPy, ni le moteur ; il lit les preuves mesurées, vérifie les durées et ne produit aucun fichier :

```sh
python3 benchmarks/plot_real_world.py --dry-run
```

Après la fin des inférences, un runtime disposant déjà de Matplotlib peut produire les exports :

```sh
python3 benchmarks/plot_real_world.py --render
```

Le rendu refuse les exécutions connues encore marquées `started`, sauf option explicite `--allow-incomplete`. Il refuse également les figures sans observation mesurée et signale les entrées manquantes. Il n’installe aucune dépendance automatiquement. Le répertoire de sortie par défaut est `research/real_world/figures/` ; seuls les graphiques dont les données existent sont exportés en SVG et PNG, accompagnés d’un manifeste des sources et des empreintes des images.

## Périmètre des deux graphiques

1. **Tokens du prompt et TTFT moteur.** Les 21 IDs principaux sont dédupliqués entre les phases initiale et reprise. Couleur et symbole identifient la phase avant ou après correctif du routeur. L’abscisse par défaut compte les IDs réellement ajoutés au cache, ce qui rend visible le coût du contexte réutilisé ; `--prompt-token-scope full` affiche le nombre de tokens du prompt complet. Les deux comptes sont conservés dans le manifeste. Le délai tracé exclut le lancement, le rendu du template et la tokenisation. Aucun ajustement causal n’est déduit du nuage de points.
2. **Secondes par token des paliers longs.** Le premier élément de `token_seconds`, qui contient le préremplissage, est supprimé. Chaque point représente une moyenne mobile sur 32 étapes complètes de décodage ; le premier point correspond donc au token 33. Ces durées d’étape excluent le callback de sauvegarde, contrairement au débit mural instrumenté du rapport. Un palier de moins de 33 tokens n’est pas transformé artificiellement en courbe. Les nombres de replis du routeur peuvent figurer dans la légende lorsque leur compteur existe ; aucune position de repli parmi les tokens n’est inventée.

Les préparations ont été vérifiées sur des données synthétiques conservées uniquement en mémoire : exclusion du préremplissage, indices des fenêtres, distinction prompt complet/ajouté, rejet des durées non finies et absence d’import graphique. Aucune courbe synthétique n’est présentée comme résultat mesuré.
