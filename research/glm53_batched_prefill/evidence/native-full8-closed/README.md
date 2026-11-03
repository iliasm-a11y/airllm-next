# Témoin CPU8 sans cache clos

Cette sélection contient les métadonnées et reçus compressés du diagnostic natif
GLM-5.3 Flash CPU10/GPU0, un modèle et un slot, trois préfills complets dans
l'ordre agent 1 → agent 600 → agent 1 et quatre échantillons greedy par requête.
Le code natif termine normalement avec 0 en 4011,844772 s ; les gardes finales
secteur et intégrité passent, sans signal de nettoyage ni croissance du swap.

Le résumé, le rapport et les reçus doivent être lus avec leurs limites :

- Les continuations terminent sur `diagnostic_budget`, sans EOG ni décision
  finale utile. Aucun objectif fonctionnel deux agents/EOS n'est qualifié.
- Le self-compare utilise le même jeu de captures des deux côtés. Son apport
  supplémentaire est le rejeu physique 0/2 avec mêmes inputs, états et sorties
  observées ; ce n'est pas un témoin CPU/GPU ou full/restore indépendant.
  Le champ automatique `comparison=full_split` ne prouve pas un tel bras.
- Trois sérialisations complètes à 208 ont 156739588 octets et le même SHA256.
  Aucune base partagée n'est créée pour réutilisation, aucune restauration
  n'est exécutée et aucun gain de cache n'est revendiqué.
- Le verrou UID est acquis tardivement par un gardien externe, environ
  517,479 s après le début du wrapper. Il n'existe pas de couverture du verrou
  avant lancement, d'héritage rétroactif de descripteur, ou de preuve complète
  d'exclusivité initiale au niveau UID.
- Les compteurs OS ne sont pas des lectures NAND ni une ventilation pure du
  calcul. Les captures sont instrumentées et ne donnent pas un gain apparié.

`selection.json` décrit chaque fichier original et gzip : tailles, SHA256,
portée et exclusions. Les 17 décompressions ont été vérifiées bit à bit.
Les fichiers `.bin`/`.f32`, poids et prompts privés complets sont exclus.
Les données binaires locales séparées sont nécessaires pour refaire toutes
les comparaisons numériques ; les seules empreintes ne les remplacent pas.
Le refus Metal ultérieur sur batterie est inclus comme reçu d'admission,
sans modèle chargé, commande GPU ni nouvelle mesure.
