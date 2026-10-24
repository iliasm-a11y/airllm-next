# Protocole AirLLM Next 5.1

Date : 24 octobre 2026. Branche de recherche, aucune fusion dans `main`.

## Révisions et périmètres

- PR #6 auditée séparément au commit exact `4ed1a58b460f8b93432e6cabc786c4ce9fc732c7`.
- Référence de débit : `ab59677`, branche `research/throughput-2026-10-24`.
- Fusion des deux historiques dans la nouvelle branche, puis correction des durées de vie et transport direct. La campagne primaire fige `cdd426e`, ses sources et son protocole avant chaque processus et après son arrêt.
- PRISME `b720462` est inspecté et ses traces réanalysées ; son ancien historique ne remplace pas les moteurs récents.
- Les commits de documentation ultérieurs doivent conserver les empreintes du moteur mesuré. Les fichiers sources réels, leurs empreintes et les commandes sont dans les preuves.

## Expérience primaire

Checkpoint local authentique : Qwen3.6-35B-A3B GGUF Q4_K_M, 23 938 321 664 octets, avec projections Q4_K/Q6_K et calcul Metal FP32. Les deux variantes utilisent exactement les mêmes poids, tokenizer, template, greedy, EOS, consignes et plafonds. Aucun poids n'est téléchargé, dupliqué, réencodé ou publié.

Trois prompts de la campagne Throughput précédente sont rejoués : réponse courte, explication et extraction depuis un contexte plus long. Deux répétitions donnent six paires, douze processus frais. Chaque paire a un ordre opposé à son autre répétition ; les six blocs sont mélangés avec la graine 20261024. Chaque processus traite une seule tâche et repart d'un état conversation/KV vide. Le plan publié avant lancement est `primary-plan.json`.

Configuration commune : cache 64 Mio, E/S 64 Mio, KV/états 256 Mio, travail 128 Mio, lecture maximale 16 Mio, tuiles 8 192 lignes, préfill par 16, contexte maximal 1 280, cache LRU, regroupement des experts déjà sélectionnés, concaténation des tuiles, un groupe SIMD par ligne. Le pipeline de tuiles est désactivé. Les poids et les équations du modèle ne changent pas.

Le candidat active trois changements : lecture packed directement dans l'allocation NumPy finale, import DLPack demandé avec fallback exact compté, métadonnées numériques groupées. Le résultat primaire mesure cette combinaison ; il n'identifie pas séparément l'effet causal de chacun. Les pilotes initiaux restent exploratoires et ne sont pas regroupés avec les répétitions primaires.

## Garde-fous

Chaque enfant est supervisé avec plafond RSS de l'arbre 2 Gio, réserve système 3 Gio, charge admise 768 Mio, augmentation du swap 128 Mio maximum et durée 600 s maximum. Pression macOS normale exigée, intervalle 50 ms, seuil d'arrêt de 32 processus possédés. Le nombre de processus et le RSS sont échantillonnés, pas garantis à chaque instant par une limite noyau. Les protections d'allocation du lecteur, du cache, des états et de MLX s'ajoutent à cette surveillance.

Le budget de buffers packed reste cache + une tuile transitoire, soit 80 Mio dans cette configuration. Lorsque DLPack retient encore une allocation à la fin d'une commande, le moteur synchronise uniquement sous pression puis revérifie le budget. Il ne force pas le GC, n'évince pas un cache utile pour masquer la pression et n'augmente pas les plafonds. Un alias encore vivant entraîne un refus avant la nouvelle lecture.

Aucune application personnelle fermée, aucun processus tiers arrêté et aucun mécanisme macOS désactivé. Un refus d'admission, un arrêt du superviseur, une mutation de source ou une sortie incomplète stoppe la campagne ; aucune relance automatique ne transforme cet événement en réussite.

## Mesures et fidélité

- Débit de décodage principal : champ de `GGUFChatSession`, entre premier token et fin du flux. Les callbacks et journaux font partie de ce temps. Agrégation par somme des tokens décodés divisée par somme des durées correspondantes.
- Débit des pas seuls : `(N-1)/sum(token_seconds[1:])`, indiqué séparément, sans le coût des callbacks entre les pas.
- TTFT moteur : préfill compris, initialisation/tokenizer/imports exclus. Préparation du tokenizer et initialisation ont leurs propres durées.
- Débit de processus complet : tokens générés, EOS inclus, divisés par durée du superviseur. Cette durée comprend démarrage Python/imports, préparation, génération et nettoyage. `worker_wall_seconds` seul exclut les imports et ne remplace pas cette mesure.
- Pour chaque forward : empreinte du vecteur complet de logits, empreinte des identités/comptes d'experts, position des états, octets logiques et compteur de lectures stockage attribué au processus. Tokens d'entrée/sortie comparés directement. Aucune capture exhaustive des valeurs de chaque banque KV n'est revendiquée dans la campagne primaire.
- RSS supervisé, pic d'allocateur MLX, minimum de mémoire disponible, swap initial/final/croissance, cache logiciel et temps de lecture packed sont conservés. Ces compteurs se recouvrent et ne s'additionnent pas en mémoire unifiée totale.
- Le profil expert séparé instrumente le helper de chargement des seules projections gate/up/down des experts, avec temps/bytes/calls de misses et hits. Il comprend préparation, cache, ownership et éventuelle synchronisation ; il ne devient pas un temps SSD ou GPU isolé.

Les résultats rapportent agrégats, médianes appariées, quartiles et étendue. Deux répétitions par prompt et trois prompts ne constituent pas une certification générale, thermique ou sémantique. L'identité des sorties conserve aussi les erreurs du modèle.

## Cache et stockage

Le cache expert de chaque processus est frais. Le cache de fichiers macOS n'est ni vidé ni contrôlé ; la machine n'est pas isolée. La campagne ne prétend donc pas mesurer un SSD physique froid. `ri_diskio_bytesread` reste un compteur macOS au niveau du processus, pas un compteur NAND. Les octets logiques et les compteurs de stockage sont présentés séparément.

Le contrôle matériel ne trouve aucun disque externe physique ; `/Volumes/SSD` est un lien symbolique vers `/`. Le travail concerne le SSD interne. Les expériences de transport synthétique utilisent de petites données générées, des lectures chaudes et des empreintes d'octets ; leur débit n'est jamais présenté comme une vitesse de génération.

## Comparaison des moteurs

La campagne locale compare deux révisions AirLLM sur le même checkpoint. L'audit concurrent fige code, licences et sources publiques dans `COMPETITORS_FR.md`. Aucun résultat Colibrì, Pulsar ou llama.cpp n'a été reproduit ici. Le checkpoint GLM-5.3 complet n'est pas disponible sur ce stockage et les grilles int4 des concurrents ne sont pas interchangeables avec GGUF Q4_K. Aucune comparaison strictement identique avec Colibrì ni victoire n'est revendiquée.
