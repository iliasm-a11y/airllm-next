# Preuves HyperStream

`raw-evidence.tar.gz` contient 106 fichiers bruts et aucun poids du modèle.
`manifest.json` donne leur taille et leur SHA-256. La restauration dans un
nouveau dossier a revérifié les 106 fichiers, puis reproduit les agrégats
indépendants de la campagne principale et ceux de l'ablation.

La campagne `primary-2` est la campagne principale complète. `primary` et les
premiers profils conservent les tentatives interrompues/refusées ; ils ne sont
pas inclus dans les agrégats de performance. `stability` et `context` sont des
essais complémentaires distincts. Le probe de 64 Mio reste une mesure du cache,
pas une certification du débit physique du SSD.

Les scripts de vérification n'ont besoin que de Python standard pour analyser
les preuves restaurées. Depuis la racine du dépôt :

```sh
mkdir -p work
mkdir work/restored-hyperstream
tar -xzf evidence-hyperstream/raw-evidence.tar.gz -C work/restored-hyperstream
python3 benchmarks/independent_hyperstream.py work/restored-hyperstream/primary-2 \
  --summary research/hyperstream/results/primary-analysis/summary.json \
  --output work/restored-hyperstream-check.json
python3 benchmarks/analyze_hyperstream_ablation.py work/restored-hyperstream/ablation \
  --output work/restored-hyperstream-ablation
```

Les chemins absolus historiques sont des données de provenance ; ils ne
modifient pas les fichiers analysés. La vérification des petits fichiers
actuels exige explicitement `--check-current-inputs` et l'ancien environnement.
Les analyses ordinaires ne lisent pas le checkpoint ni le GPU.
