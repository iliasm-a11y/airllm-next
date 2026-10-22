# Validation matérielle à effectuer sur le MacBook Air M5

Aucune étape de ce protocole n’a été exécutée sur M5 dans la livraison cloud.

## Avant les grandes lectures

Installer avec `scripts/setup_next_macos.sh`, puis lancer
`scripts/validate_next_macos.sh`. Les tests fabriquent des GLM miniatures sans
download. Les tests des modèles Next sont exécutés avec Metal et comparés à
Transformers CPU. Examiner les erreurs numériques et les tests ignorés avant
une génération complète. Le test d’import et les mesures d’E/S restent distincts.

Les versions testées ici sont MLX 0.32.4 et Transformers 5.20.2. Sur M5, macOS
26.2+ est requis pour bénéficier des Neural Accelerators via MLX selon Apple.
Il n’est pas établi que chaque forme et précision du présent backend les
emploiera ; le vérifier dans une capture réelle.

## Préparation et stockage

Avec les poids sur le SSD externe, exécuter `airllm-next prepare CHEMIN`, puis
`airllm-next tune CHEMIN`. Conserver le modèle sur le même volume pendant les
comparaisons. Les résultats répétés peuvent venir du cache OS ; un `mmap` ouvert
ne prouve aucune lecture du SSD. Le compteur Linux `/proc/self/io/read_bytes`
n’existe pas sur macOS : il sera `null`, pas remplacé par les octets logiques.
Utiliser les outils macOS de suivi du disque pour une mesure de volume physique,
en précisant qu’un compteur de volume peut inclure d’autres processus.

Mesurer séparément : index non préparé, index préparé après nouveau processus,
processus et poids en cache, premier prefill, premier token, puis tokens de
décodage. Un processus démarré et un index prêt ne sont pas une réponse prête.

## Quantification et préchargement

Commencer par une conversion bornée `convert ... --max-matrices 10`. Contrôler
la place restante sur le volume avant la conversion complète. Comparer un
corpus fixe à une exécution de référence réalisable sur une machine disposant
de suffisamment de mémoire ; relever pertes/NLL, divergences de logits,
accord des tokens et tâches applicatives représentatives. Les fixtures
aléatoires de cette archive ne certifient pas la qualité linguistique de GLM.

Pour calibrer le vrai coût du préchargement :

```bash
.venv-next/bin/airllm-next calibrate /Volumes/SSD/GLM-5.3 \
  --prompt "Explique une boucle de programmation" --max-new-tokens 8
```

Cette commande fait plusieurs générations réelles et peut être très lente sur
744B. Elle ne télécharge rien. La décision ABBA et le replay de cache sont
persistés. Ajouter `--quantized-dir ...` pour calibrer le dérivé exact.

## Mesure soutenue et Metal

```bash
.venv-next/bin/python benchmarks/sustained_next.py /Volumes/SSD/GLM-5.3 \
  --seconds 300 --tokens 32 --device gpu --profile balanced
```

Répéter avec `--profile performance`. Comparer première et dernière minute,
latences et mémoire, pas seulement la meilleure seconde. Un token lent peut
dépasser la durée demandée : la limite de durée est vérifiée entre requêtes.
Le SSD Fanxiang n’est pas présumé fonctionner à un débit annoncé tant que le
modèle exact, le câble, le boîtier et le port n’ont pas été mesurés ensemble.

Une capture GPU réelle peut être enregistrée :

```bash
MTL_CAPTURE_ENABLED=1 .venv-next/bin/python benchmarks/profile_metal_next.py \
  /Volumes/SSD/GLM-5.3 --capture evidence-next/decode.gputrace
```

Inspecter la trace avec Xcode : temps de kernels, synchronisations, copies,
occupation et allocations. Les métriques GPU et températures non disponibles
restent à `null` dans les rapports ; aucune approximation n’est déguisée en
mesure matérielle. Une capture Metal ne valide ni le contexte maximal ni toute
la qualité du checkpoint complet.
