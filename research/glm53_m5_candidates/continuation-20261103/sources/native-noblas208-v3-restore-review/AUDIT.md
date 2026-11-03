# Revue de restauration noBLAS v3 — nouvelle copie non exécutée

Le comparateur original sous `work/native-noblas208-v3-candidate` reste gelé et
inchangé pendant le modèle actif. Cette copie séparée n’est pas une nouvelle
version mesurée du helper. Aucun test, build, import natif, modèle, GPU ou
comparaison de payload n’a été exécuté pour cette revue.

L’original exige déjà le vrai pont CPU8 → full v3 **complet**, par son appel à
`v.compare` (`restore_compare.py:38–43`). Les deux appels `arm_dependencies`
aux lignes 67–68 vérifient chacun les propres clés, paramètres/sources/binaries
v3 et les observations réelles du registre, ainsi que le smoke réel du build.
Le paramètre `cpu_accelerator_policy` et la politique/version 3 ne sont donc
pas remplacés par une simple étiquette v2 héritée. Les libellés v2 restants
sont trompeurs pour la lecture, mais ne relâchent pas ces contrôles effectifs.

`_read` exige trois requêtes closes, gardes claires, IDs/positions complets et
SHA des fichiers bruts. L’original compare le snapshot à 208, les logits entiers,
les états finaux sérialisés, les tenseurs du suffixe avec routes ordonnées,
les écritures observées dans leur ordre, les quatre continuations diagnostiques
ou le même EOS et les replays 0 → 2. Un résultat limité à 128 tokens ne satisfait
pas ce chemin. Aucune preuve early128 n’est admise comme bras complet.

Le défaut concret est l’absence de lecture de `state_checkpoints.jsonl` dans
le bras **restore**. `geometry.direct_restore`, appelé à la ligne 74 de
l’original, lit seulement les checkpoints du full. Le verdict des lignes
118–121 ne dépend donc pas des checkpoints intermédiaires restore : une ligne
256/384/512 manquante ou divergente pourrait être ignorée si les autres critères
et la fin concordent. Les écritures observées déjà comparées ne remplacent pas
la comparaison de toutes les sérialisations opaques promises.

La nouvelle copie est bornée au profil noBLAS **32** et épingle les sources
originales v3. Elle ajoute :

- L’ensemble exact des checkpoints correspondant aux fins des vrais appels
  decode après validation de leur géométrie. Le restore 0 comprend l’unique
  préfixe commun à 128/208 ; les restores 1/2 émettent seulement leurs checkpoints
  de suffixe. Aucun nouveau checkpoint 208 fictif n’est exigé de ces deux bras.
- La comparaison de chaque checkpoint restore à son homologue full, sa liaison
  finale au fichier brut déjà vérifié, et le replay des checkpoints de suffixe
  0 → 2.
- L’identité des observations de registre full, restore et smoke réel du build,
  après leur validation indépendante par les sources v3.
- La liaison de `input_tokens`, `batch_index` et `phase` aux reçus de batch
  propres à chaque bras. L’index du restore 0 continue après le préfixe commun ;
  ceux des restores 1/2 commencent à zéro. Les différences full/suffixe ne sont
  pas normalisées aveuglément entre bras.
- Le payload canonique à 208 fixé à 156 739 588 octets et
  `4c4d6f24b2cf3c4f30a7894b8609035960a3ec5687521c942aa6a578e2e5a3cd`,
  et des libellés v3 explicites dans le nouveau rapport.

Tous les critères numériques CPU restent bit à bit, routes/tokens ordonnés
exacts et valeurs flottantes capturées finies. Les snapshots/checkpoints opaques
sont comparés par taille/SHA ; aucune finitude de leurs octets n’est inventée.
Le helper gelé n’exporte pas de fichier brut du préfixe : son SHA de snapshot
complet en mémoire est confronté aux checkpoints full à 208. Les fichiers bruts
finaux et post-continuation sont, eux, relus/hachés par le comparateur d’origine.

`test_restore_review.py` contient des fixtures préparées de refus de couverture,
divergence intermédiaire, métadonnées de callback et admission complète. Elles
ne sont pas exécutées pendant le modèle actif. Une contre-revue en lecture seule
par campaign_harness n’a pas trouvé de nouveau P1/P2 matériel dans cette copie.
Ce constat source ne qualifie ni B, ni EOS utile, ni RONIN C, ni la performance.
Un futur essai restore32 reste conditionné au full32 complet strictement positif
et aux gardes habituelles ; aucun essai n’est déclenché ici.
