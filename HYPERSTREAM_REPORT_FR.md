# AirLLM Next HyperStream — résultats vérifiés sur M5

27 octobre 2026. Recherche sur une branche dédiée issue de `ca3a429`, sans fusion dans `main`.

## Résultat réel

Le checkpoint Qwen3.6-35B-A3B GGUF de **23 938 321 664 octets** est exécuté par AirLLM natif sur un **MacBook Air M5 de 16 Gio**, avec Metal. Aucun service Ollama ne génère les réponses. Son fichier local est lu directement.

Sur huit processus frais, deux prompts et deux répétitions équilibrées, le décodage passe de **1.474 à 1.584 token/s (+7.47 %)**. Le débit du processus complet passe de **0.845 à 0.923 token/s (+9.21 %)**. Les quatre paires sont favorables pour ces deux mesures.

Les huit réponses atteignent EOS. Entrées, sorties, textes, empreintes des vecteurs complets de logits, routes et positions sont identiques par paire. Un vérificateur indépendant sans import du moteur ni de l'analyseur principal retrouve les agrégats et contrôle les sources, commandes et protections. Les réponses intégrales sont conservées sans réécriture.

**Les objectifs de 5, 10 et 20 tokens/s ne sont pas atteints. Ces résultats concernent ce Qwen et ces conditions ; ils ne valident pas GLM complet ni des modèles de centaines de gigaoctets.**

## Changements implémentés

1. **Historique du cache borné.** L'ancienne purge dépendait de la somme hits/misses/évictions : un scan pouvait sauter indéfiniment sa condition. Un compteur de demandes déclenche désormais la purge toutes les 8 192 requêtes. Les observations utiles à la demande courante et aux valeurs résidentes sont conservées. Le compteur n'est pas remis à zéro lors d'une purge sous pression.
2. **Parsing GGUF groupé et borné.** `string_buffer_bytes=65536` regroupe les lectures des tableaux de chaînes dans au plus 64 Kio, sous le budget existant. La borne dérivée des longueurs restantes empêche de lire les poids par anticipation. Le buffer est libéré à la fin du tableau et sur erreur ; les lecteurs partiellement initialisés se ferment correctement. La valeur par défaut reste zéro.
3. **Chevauchement exact des experts.** `expert_pipeline=True` soumet la projection gate des experts déjà choisis, prépare les poids up pendant ce calcul, puis attend et calcule up. Il n'y a aucune prédiction remplaçant le routage. Une seule projection reste en attente. Le budget simultané couvre buffers, imports et sorties ; les cas trop grands ou sous pression reviennent au chemin synchrone. Les erreurs drainent les commandes avant de libérer les propriétaires. L'option reste désactivée par défaut.
4. **Admission entre processus.** Une option de campagne attend une fenêtre normale stable avant chaque lancement. Le superviseur enfant refait sa propre admission. Aucun seuil RSS, réserve, pression ou swap n'a été diminué.

Le format Q4_K/Q6_K, les poids, le routage, les kernels et l'ordre de réduction par expert sont conservés. Aucune nouvelle quantification, suppression d'experts, distillation ou conversion du checkpoint n'est introduite. Le fichier a été ouvert en lecture seule et son SHA-256 complet a été vérifié avant la campagne finale puis après les générations : `f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`.

## Audit et chemin critique

Le profil instrumenté de la référence termine sur la réponse complète « 42 ». Il compte 19,08 s sous cProfile, dont 9,41 s dans le chemin groupé d'experts, 6,29 s cumulées dans `preadv` et 2,31 s dans les deux parsings GGUF. Ces coûts se recouvrent : ils ne s'additionnent pas. cProfile ajoute du coût ; ces durées ne sont pas le benchmark principal ni une mesure pure du SSD ou du GPU.

Le code possède déjà des budgets séparés pour les poids, E/S, KV/états, travail et l'allocateur MLX. HyperStream réutilise ces mécanismes et leur réserve système. Un triple buffering général ou une prédiction multicouche n'est pas activé : le routage futur dépend du calcul, et la variante mesurée ne nécessite ni threads supplémentaires ni lecture d'experts inutilisés.

## Mesures de démarrage et mémoire

| Mesure | Référence | Candidat |
|---|---:|---:|
| Initialisation médiane, tokenizer inclus | 1.343 s | 0.708 s |
| Préparation du tokenizer médiane | 0.927 s | 0.493 s |
| TTFT moteur médian | 17.24 s | 15.92 s |
| Pic RSS échantillonné du processus | 417.25 Mio | 460.88 Mio |
| Pic allocateur MLX | 156.27 Mio | 156.27 Mio |
| Minimum disponible système | 3.842 Gio | 3.996 Gio |
| Croissance maximale du swap | 0 | 0 |
| Lectures logiques par token décodé | 1.964699 Go | 1.964699 Go |

Le candidat utilise **43,63 Mio de RSS supplémentaire** au pic observé. Le gain du microbenchmark des métadonnées du cache ne doit pas être présenté comme une baisse de la mémoire totale du modèle. RSS, MLX et cache de fichiers macOS sont des mesures qui se recouvrent.

Le TTFT moteur exclut imports, tokenizer et initialisation ; il inclut préfill et calcul du premier token. Le préfill isolé n'est pas mesuré. Le débit de processus comprend imports, préparation, génération et supervision, mais exclut l'attente préalable d'admission, d'environ cinq secondes par essai ici. Le décodage compte les tokens après le premier, EOS compris, et inclut restitution et journalisation entre pas. Chaque variante produit 144 tokens, dont 140 après les premiers tokens ; durées de décodage 95.010 / 88.402 s. Le facteur médian apparié est 1.0724, avec étendue 1.0360–1.1132. Quatre paires restent un petit échantillon descriptif.

### Composants isolés

Parsing seul, quatre répétitions AB/BA, mêmes métadonnées et index : **0,376 → 0,157 s** sans conserver le tokenizer et **0,788 → 0,313 s** en conservant ses métadonnées. Les appels de lecture passent respectivement de 504 169 à 8 446 et de 1 000 111 à 8 489. Aucun poids n'est lu. Lorsque le tokenizer est omis, la lecture groupée demande davantage de ses octets, environ 10,01 contre 4,04 Mo ; elle économise des appels, pas du trafic logique.

Microbenchmark synthétique LRU de deux octets, 200 000 demandes uniques : observations de fréquence/coût **200 000 → 3 395**, pic Python tracé **30 251 451 → 1 729 553 octets (−94,28 %)**. Absences, évictions et valeurs résidentes concordent. Ce test n'est ni une mesure RSS ni une génération modèle.

L'ablation supplémentaire conserve exactement le même code, le buffer de chaînes, les compteurs du cache et les mêmes budgets ; elle ne change que le pipeline gate/up. Ses deux paires passent de **1.549 à 1.586 token/s (+2.36 %)** ; les deux ratios appariés sont favorables. Les tokens, logits, routes et octets logiques sont identiques. Ce petit test d'attribution confirme un gain modeste du pipeline et ne lui attribue pas tout le gain de la combinaison principale. Les résultats sont fournis dans `research/hyperstream/results/ablation-analysis/`.

## Sortie longue, contexte et stockage

Une réponse supplémentaire complète atteint **318 tokens et EOS**, avec **1.565 token/s**, TTFT **24.45 s**, génération **227.02 s**, pic RSS **449.53 Mio**, pression normale et aucune croissance du swap. Ce test d'environ quatre minutes ne constitue pas une validation thermique de trente minutes ni une certification d'absence de toute fuite.

Un autre essai fournit **364 tokens d'entrée** et termine avec la réponse complète « 42 ». TTFT **110.70 s**, durée de génération **112.16 s**, RSS **417.55 Mio**, aucune hausse du swap. Le préfill reste lent. Le contexte configuré de 1 280 tokens n'est pas une validation d'une conversation réelle de cette longueur.

Les tests numériques Metal couvrent Q4_K/Q6_K, DLPack avec/sans import direct, batch 1/3 et préfill MoE irrégulier. Ils comparent le chemin synchrone et asynchrone. Les réponses du modèle ne sont pas un audit général de qualité : la réponse longue emploie notamment l'expression incorrecte « architecture à experts moindres », conservée dans le brut.

Le stockage vérifié est le **SSD interne APPLE SSD AP0512Z**. Aucun disque physique externe n'est énuméré ; `/Volumes/SSD` est un lien vers `/`. Un essai distinct, douze processus frais lisant chacun 64 Mio par blocs de 1 Mio, retrouve tous les hashes, mais les compteurs OS indiquent **zéro octet de stockage imputé au processus** : les quelque 17–20 Gio/s logiques reflètent le cache, même avec `F_NOCACHE`. Ce résultat négatif est conservé et ne certifie pas le débit du SSD.

Dans la campagne modèle, les compteurs OS du processus donnent environ **210,00 / 199,04 Mo par token décodé**, contre **1,964699 Go logiques** dans les deux variantes. Le cache de fichiers n'est pas contrôlé. Il ne s'agit ni de trafic NAND, ni d'une preuve de réduction physique causée par le pipeline, ni d'un trajet direct SSD→GPU.

Une lecture séquentielle intégrale distincte après les générations lit 23,94 Go en **5,45 s cumulées dans les appels read (4,39 Go/s logiques)**. Le compteur OS du processus augmente de **22,97 Go**. Le hash complet, calculé hors des durées de lecture isolées, porte le total à **12,52 s**. La politique `F_NOCACHE` est appliquée au seul descripteur du test ; il n'y a aucune purge. Cette mesure vérifie un volume de stockage substantiel sur le SSD interne, sans certifier la bande passante NAND ni une condition froide contrôlée.

## Validation, conservation et limites

Dernière suite : **1324 passed, 4 skipped, 26 subtests passed in 20.11s**. Les tests Metal ciblés ont aussi donné 54 réussites. Le rapport brut, le JUnit et les gardes accompagnent les résultats. Les protections conservent réserve 3 Gio, RSS 2 Gio, croissance swap 128 Mio, pression normale et échantillonnage 50 ms. Les 53,74 Mo de swap préexistants de la campagne principale sont affichés ; ils n'ont pas augmenté.

Les premiers essais arrêtés sont conservés : un profil interrompu sous pression mémoire, un refus avant lancement, une attente sans fenêtre suffisante et une campagne stoppée après sa référence. Ils ne sont pas mélangés aux huit mesures complètes. La vérification indépendante rejette aussi campagne inachevée, garde relâchée, logits/source/commande modifiés, réponse tronquée et valeur NaN.

Le parser est testé avec un fichier sparse de plus de 1 Tio et des offsets 64 bits sans lecture des poids. C'est une validation d'indexation, pas l'exécution d'un modèle de 1 Tio. Les familles GLM, autres Transformers, SSD externe, démarrage réellement froid, utilisation GPU attribuée au processus, temps Metal pur, contextes très longs et stabilité quotidienne restent non qualifiés par cette mission.

Le code et les données sont sur une branche de recherche dépendant de la branche 5.1 existante. Les formats et chemins historiques sont conservés ; les optimisations de lecture et de pipeline restent facultatives. Une pull request en brouillon permet la revue. Aucun changement de `main`, fusion ou nouvelle version stable n'est effectué. Les commits locaux et les commits créés par le connecteur GitHub ont des SHA différents ; leurs arbres Git ont été comparés exactement. La correspondance figure dans `research/hyperstream/PUBLICATION.json`.

Voir `research/hyperstream/README.md` pour reproduire les tests et `research/hyperstream/results/` pour les résumés, CSV et réponses. L'archive de preuves brutes est accompagnée d'un manifeste SHA-256 ; aucun poids du modèle n'y figure.
