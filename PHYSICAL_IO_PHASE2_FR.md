# Phase 2 — chemins de lecture et E/S physiques

Le protocole distingue désormais les octets logiques copiés, l'activité de stockage imputée au processus et les compteurs du périphérique. Les essais physiques ci-dessous concernent **le SSD interne APPLE SSD AP0512Z**. Le Fanxiang PS1008 2 To n'était pas connecté : **aucune validation de ce SSD externe, de Thunderbolt ou d'un débit NAND**.

## Changements

- `pread_nocache` est un mode explicite macOS qui demande `F_NOCACHE` sur un descripteur ouvert séparément. Le cache de descripteurs sépare les politiques ; une éviction ne ferme pas le descripteur emprunté par une lecture en cours. Une plateforme ou un appel système incompatible provoque une erreur, sans repli silencieux.
- Les régions macOS jusqu'à 8 Mio utilisent un `pread` direct lorsque la première lecture est complète. Cela évite le remplissage initial d'un `bytearray` et sa copie ; les lectures partielles et interruptions conservent leur traitement exact et borné. Les régions plus grandes conservent `preadv` lorsqu'il existe. Les vues retournées peuvent être en lecture seule, comme avec `mmap`.
- Le profiler utilise sur macOS `proc_pid_rusage(RUSAGE_INFO_V2).ri_diskio_bytesread`. Linux conserve `/proc/self/io:read_bytes` ; l'indisponibilité reste `null`. L'ABI publique installée a été vérifiée par compilation : structure 160 octets, champ de lecture à l'offset 144, flavor 2 ([preuve](evidence-io-phase2/physical-counter-abi.json)).
- Le transport compressé `lossless_dir` refuse explicitement `pread_nocache`, car son lecteur de packs n'implémente pas cette politique.

## Protocole exécuté

`benchmarks/benchmark_physical_io.py` crée dans un répertoire temporaire qu'il possède un safetensors pseudo-aléatoire de 128 Mio. Il conserve seulement son manifeste/hachage et supprime la charge après l'expérience. Chaque lecture sélectionne une ligne de 1 Mio ; l'ordre des 128 lignes est mélangé mais identique pour toutes les variantes. Les SHA-256 de chaque ligne et de leur séquence doivent correspondre exactement au manifeste indépendant.

Chaque travail s'exécute dans un nouveau processus. Les révisions, modes et scénarios sont mélangés dans chacun des 5 blocs de répétition. La durée de lecture exclut le calcul des hachages. Les compteurs de lectures logiques et d'appels sont échantillonnés après tout échauffement ; chaque résultat mesuré porte exactement sur 134 217 728 octets et 128 appels.

Deux scénarios sont séparés :

1. Après création et travaux précédents : état du cache OS **non contrôlé**, jamais qualifié de froid.
2. Après lecture intégrale explicite par `pread` normal : échauffement déclaré, distinct des lectures mesurées.

La deuxième série crée sa propre charge en demandant `F_NOCACHE` au **descripteur d'écriture**, puis effectue `flush` et `fsync`. Elle exécute uniquement les deux variantes de lecture nocache, sans échauffement normal. Aucune purge privilégiée, pression mémoire artificielle, écriture sur disque brut ou modification de fichiers existants.

La phase 1 ne connaît pas l'énumération nocache : pour cette seule comparaison, le banc applique `F_NOCACHE` au descripteur de son lecteur dédié. Sa boucle de lecture demeure inchangée ; chaque ligne déclare `baseline_nocache_descriptor_harness=true`.

## Résultats observés

Les **70 processus ont rendu des hachages exacts**. Les sources de lecture et de comptage ont un SHA-256 stable et enregistré pour chaque travail. Les valeurs sont des médianes sur 5 essais ; les journaux contiennent chaque essai.

| Série / mode | Phase 1 | Phase 2 | Rapport phase 1 / phase 2 | Compteur processus |
|---|---:|---:|---:|---|
| Échauffement explicite, `pread` | 7,078 ms | 6,354 ms | 1,114× | 0 octet sur chaque essai |
| Cache non contrôlé, `pread` | 7,580 ms | 7,021 ms | 1,080× | 0 octet sur chaque essai |
| Échauffement explicite, `mmap` | 12,792 ms | 12,764 ms | 1,002× | 0 octet sur chaque essai |
| Écriture nocache, lecture nocache | 54,219 ms | 53,216 ms | 1,019× | 131 596 288 à 135 532 544 octets par essai |

Les premières lignes mesurent des chemins servis sans hausse du compteur de stockage du processus. Leurs taux logiques, environ 18 000–20 000 Mio/s, **ne sont pas des débits SSD**. Dans les résultats JSON, le nom exact est `logical_mib_per_read_second` (Mio/s).

Pour la série créée nocache, `disk0` progresse de 131 596 288 à 135 540 736 octets selon l'essai, à proximité du compteur processus. Cela constitue une observation de lectures réelles imputées par l'OS au stockage interne. Les compteurs ont des portées différentes et peuvent inclure la réadahead, des métadonnées ou une activité extérieure ; ils ne garantissent pas une attribution octet pour octet au fichier et ne mesurent pas les cellules NAND. Le gain médian de 1,019× sur ces 5 essais physiques est trop faible pour revendiquer une accélération robuste du disque.

La pointe RSS du banc reste inférieure à 38 Mio, imports et métadonnées compris. La charge temporaire maximale est de 128 Mio par série, et le tampon de sélection est de 1 Mio. Ces chiffres décrivent ce banc, pas la consommation d'un modèle.

## Journaux et reproduction

- [Série comparée complète : 60 résultats](evidence-io-phase2/physical-paired-128m/report.json)
- [Série avec lectures de stockage observées : 10 résultats](evidence-io-phase2/physical-nocache-created-128m/report.json)
- Dans chaque répertoire : `rows.jsonl`, `fixture-manifest.json`, `worker-stderr.log`.

```sh
python benchmarks/benchmark_physical_io.py \
  --output evidence-io-phase2/reproduction-cache \
  --baseline-source /chemin/vers/snapshot-phase1 \
  --size-mib 128 --chunk-kib 1024 --repeats 5

python benchmarks/benchmark_physical_io.py \
  --output evidence-io-phase2/reproduction-storage \
  --baseline-source /chemin/vers/snapshot-phase1 \
  --size-mib 128 --chunk-kib 1024 --repeats 5 \
  --modes pread_nocache --fixture-write-policy nocache \
  --scenarios uncontrolled_after_creation_and_prior_jobs
```

Le répertoire de sortie doit être nouveau. Pour tester le Fanxiang réellement connecté, fournir `--fixture-parent /Volumes/volume-reel` et vérifier la chaîne `fixture_disk_identity` enregistrée. Un lien symbolique vers `/` n'est pas une preuve de disque externe. La limite de charge est 256 Mio, les lectures 8 Mio au maximum par sélection du banc, chaque travail expire après 120 secondes.

Les 39 tests ciblés réussissent : isolation des politiques et éviction concurrente, refus sans fuite de descripteur lors d'un `fcntl` échoué, véritables lectures macOS nocache, lectures courtes et interruptions, bornes et exactitude du protocole, séparation des compteurs d'échauffement, erreurs de transport, profilage et fiabilité du stockage.

## Portée des API

Sources primaires : [API Apple du champ `ri_diskio_bytesread`](https://developer.apple.com/documentation/kernel/rusage_info_v2/1577572-ri_diskio_bytesread), [définitions publiques XNU](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/resource.h), [constantes publiques `F_NOCACHE`](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/fcntl.h), [implémentation de lecture XNU](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/vfs/vfs_cluster.c).

`F_NOCACHE` demande une politique de cache et ne purge pas tout le cache OS. Les 60 essais de la première série ont notamment observé **zéro** au compteur processus, y compris en mode nocache sur une charge déjà écrite/cachée. Une valeur nulle ne prouve pas que tous les octets venaient du cache ni qu'aucun autre processus n'accédait au disque. Une valeur positive ne prouve aucun débit Fanxiang, aucune validation de modèle géant et aucune performance thermique durable.
