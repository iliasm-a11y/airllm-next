# Preuves M5 HyperStream Recovery — livraison partielle

C5 conserve 88 workers scellés, 22 expériences acceptées, une exclusion TensorOps (4 emplacements), quatre workers de spéculation libre profondeur 16 manquants et quatre emplacements de combinaison bloqués. Aucune clôture de campagne complète n’est annoncée.

- `final/Verification-independante-partielle-finale.json` : audit indépendant des 88 jobs et 22 résultats disponibles, sans nouveau modèle ni modification de C5.
- `final/Verification-independante-source-finale.json` : quatre nouveaux workers courts AB/BA sur la révision de contrôle finale, après préparation distincte ; aucun rejeu des 100 jobs sur cette révision.
- `final/Provenance-source-finale-M5.json` : correspondance entre commits locaux et arbres Git publics, 524 tests CPU réussis et contrôles numériques conservés.
- `gpu/` et `gpu-final-source/` : grilles GPU réelles et tenseurs bruts. Trois répétitions de 160 cas produisent 480 exécutions au total ; 160 cas nominaux distincts. Reuse bit à bit ; TensorOps globalement non qualifié.
- `final/Resume-preuves-finales-M5.json` décrit l’archive brute locale sélectionnée et donne les SHA du ZIP et du manifeste complet local : quatre workers longs H1/H2 et quatre courts source finale avec leurs tableaux FP32, gardes, hashes et sources retenues. L’archive n’est pas téléversée dans Git ; cette sélection publique JSON ne remplace pas tous les tableaux bruts locaux de C5.
- `router/` : preuve numérique FP32 compacte et vérificateur autonome conservés.
- `cleanup/` et les deux bilans finaux : 512 SIGTERM tracés, zéro SIGKILL et zéro fichier supprimé par le nettoyage. Les inventaires privés de processus ne sont pas publiés.

Le rapport français à la racine distingue préfill, décodage, durées requête/worker/processus, lectures logiques, compteurs OS et limites non mesurées du NAND. Aucun résultat de copie n’est extrapolé à la génération libre.
