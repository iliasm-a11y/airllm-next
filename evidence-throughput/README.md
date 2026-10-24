# Preuves Throughput

`raw-json-logs.tar.gz` conserve sans perte les JSON, journaux et commandes des expériences. `raw-manifest.json` contient les empreintes SHA-256 des originaux et de l'archive, ainsi que celles des captures binaires conservées localement. Aucun poids ni document personnel n'est inclus.

Restaurer dans un dossier neuf :

```sh
python benchmarks/restore_throughput_evidence.py --output /nouveau/dossier/preuves
```

La campagne principale est dans `primary-balanced/`. Le premier plan non équilibré et son interruption volontaire sont dans `primary/`. Les contrôles courts par retrait sont dans `ablations/`. Les pilotes, profils, contrôles d'états, hash intégral du modèle, deux inférences précédentes et journaux de tests restent distincts.

Les tableaux lisibles et calculs reproductibles sont publiés dans `research/throughput/results/` et le rapport `THROUGHPUT_REPORT_FR.md`. Le script d'analyse refuse des paires incomplètes, des violations et des divergences d'empreintes ; l'étude du draft n'est pas une mesure de génération spéculative.
