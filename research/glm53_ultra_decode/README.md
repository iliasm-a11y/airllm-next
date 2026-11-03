# GLM-5.3 Flash — recherche de décodage M5

Cette branche ajoute des kernels et de vrais tests locaux, mais **ne livre pas
un moteur GLM complet plus rapide et qualifié**. Les nouveaux composants
résidents sont fidèles sur les cas testés ; CPU10 les exécute plus vite que
Metal, coûts de lancement et synchronisation inclus. La génération complète
contemporaine est refusée avant lancement par les réserves mémoire/disque.

Voir [le rapport français](RAPPORT-FR.md), les
[preuves sélectionnées](evidence/selection.json) et le
[résumé des composants](evidence/component-summary.json).

## Périmètre et isolation

- Départ : PR #23 `6bdeec357154a7740833b4ad694d08e02aef2f96`, vérifiée sur GitHub
  avant création de `research/glm53-ultra-decode-engine`.
- Aucun fichier du préfill, de RONIN ou du runtime de production n'est modifié.
  Un workflow de tests indépendant est ajouté.
- Poids, quantification et sélection des experts inchangés. Les microtests
  utilisent des poids **synthétiques dans les formats natifs**, sans lire le
  checkpoint. Ils ne démontrent aucune fidélité globale GLM.
- `scheduler.py` est un prototype de politique sans branchement au moteur :
  désactivé par défaut, il refuse la promotion d'une preuve de composant en
  preuve de modèle. Aucun placement automatique n'est activé.

## Code livré

| Fichier | Rôle |
|---|---|
| `decode.metal` | Q8_0 et Q2_K, un token, tuiles de1/2/4 lignes ; deux banques originales possibles dans un lancement |
| `native_component.cpp` | Graphe CPU persistant et un seul pool partagé pour une paire ; poids empruntés sans copie |
| `decode_timer.mm` | Buffers/pipelines persistants, quantification CPU native, temps GPU et temps total synchronisé séparés |
| `components.py`, `guard_components.py` | Génération synthétique bornée, comparaison au graphe et au dot ARM, surveillance des processus possédés |
| `native_decode.cpp`, `timing.inc`, `residency.inc` | Auxiliaire compilé CPU10/slot1/batch8, temps par appel, compteurs OS, pages et captures optionnelles |
| `model_profile.py` | Admission et supervision bornées du profil CPU contemporain ; aucun abaissement des seuils en CLI |
| `profile_analysis.py`, `historical_analysis.py` | IntervallesN−1, regroupements par opération/couche et bornes conditionnelles explicites |
| `test_decode.py` | Contrats de refus, intégrité des intervalles et politique CPU/GPU |

Le profil natif garde un contexte512 et batch/ubatch8 pour rester comparable à
la référence de diagnostic antérieure. Il refuse `prompt_tokens+budget>512`.
Il ne remplace pas les sessions RONIN1024 et leurs prompts Max complets.
Le mode `--diagnostic` sérialise les callbacks et écrit des tenseurs ; son débit
ne doit jamais être présenté comme une performance de production. Sans cette
option, les temps `llama_decode` excluent les copies de captures, le préfill et
le premier token ; ils ne constituent pas une mesure de latence d'une réponse
RONIN de bout en bout.

## Reproduction

Depuis la racine du dépôt, avec Python disposant de NumPy, psutil et pytest :

```sh
export PYTHONPATH=.:air_llm
python -m research.glm53_ultra_decode.build \
  --source /chemin/llama.cpp-3f8af15 \
  --runtime /chemin/runtime-v4/bin \
  --output /chemin/nouveau-build

python -m research.glm53_ultra_decode.guard_components \
  --runtime /chemin/runtime-v4/bin --build /chemin/nouveau-build \
  --output /chemin/nouvel-essai-Q2 --format 2 --rows 2048 \
  --width 4096 --banks 2 --repetitions 21

python -m research.glm53_ultra_decode.guard_components \
  --runtime /chemin/runtime-v4/bin --build /chemin/nouveau-build \
  --output /chemin/nouvel-essai-Q8 --format 8 --rows 1536 \
  --width 4096 --banks 1 --repetitions 21

python -m research.glm53_ultra_decode.model_profile \
  --model /chemin/GLM-5.3-Flash-Q2_K_S \
  --receipt /chemin/recu-SHA256-actuel.json \
  --runtime /chemin/runtime-v4/bin --probe /chemin/nouveau-build/native-decode \
  --output /chemin/nouveau-profil-CPU32 --tokens 32 --diagnostic

python -m pytest -q research/glm53_ultra_decode
```

Tous les dossiers de sortie doivent être nouveaux. Répéter avec une seconde
graine ; aucune répétition n'est une nouvelle trajectoire de modèle.
Le modèle complet exige pression normale, secteur/capot/horloges valides,
5,5 Gio disponibles et32 Gio libres sur le volume de sortie. Les composants
réservent3 Gio au système et256 Mio au payload estimé. Les deux gardent RSS8,5 Gio
et croissance du swap global32 Mio. Les gardes sont échantillonnées ; une crête
entre deux échantillons peut rester invisible. Le swap global n'est pas
attribué exclusivement au modèle.

Pour une campagne séquentielle, relever le swap utilisé avant le premier essai
et transmettre la même valeur en octets à chaque commande via
`--campaign-swap-used-bytes`. Ce contrôle réduit le budget restant des essais
suivants et refuse une croissance cumulée supérieure à32 Mio. Une baisse du
swap ne relève jamais le plafond original de32 Mio par essai. Les réserves
mémoire et disque restent identiques ; aucune option ne les abaisse.

Les nouvelles options de capture et bibliothèques doivent repasser leur propre
comparaison logits/routes/états avant toute utilisation pilote. Les mesures
historiques des PR17–23 ne qualifient pas automatiquement ce nouvel auxiliaire.

## Suite de la campagne

1. Rendre les réserves disponibles sans supprimer de données ni fermer
   d'applications personnelles automatiquement.
2. Établir un CPU contemporain complet, capturé puis non instrumenté.
3. Comparer32/64/128 sorties et une sortie naturelle EOS, avec les mêmes
   poids/prompts/paramètres. Exécuter A/B/B/A et arrêter une campagne au premier
   refus plutôt que d'en exploiter les seuls passages réussis.
4. N'intégrer un kernel ou cache que si son gain total et les seuils existants
   de logits/routes/états sont satisfaits. Les coûts doivent inclure copies,
   quantification, synchronisations et pression mémoire.
5. Alors seulement, mesurer deux réponses utiles d'agents Max distincts,
   séquentielles sur le slot partagé. La charge durable de600 agents reste une
   qualification séparée.
