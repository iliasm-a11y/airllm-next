# AirLLM Next — deuxième phase expérimentale, 23 octobre 2026

Cette phase distingue les lectures logiques, le service du cache OS et les
lectures de stockage comptabilisées par macOS. Elle ajoute une politique de
lecture explicite, corrige des coûts du prédicteur et introduit un backend MoE
Granite expérimental à chargement sélectif. Les résultats positifs et négatifs
ainsi que les essais préliminaires sont conservés.

**Le Fanxiang PS1008 2 To est absent.** Les mesures physiques portent sur le
SSD **interne APPLE SSD AP0512Z**, sur ce MacBook Air M5 de 16 Go. Aucun résultat
n'est une validation du Fanxiang, d'un lien Thunderbolt, d'un modèle géant ou
d'un comportement thermique durable.

## Provenance et état de publication

Référence de départ : `3e4138a1a6a20decf5c9be187fd5062836cd1caf`, tête vérifiée
de `research/io-wall-2026-10-23` et de la [PR brouillon #1](https://github.com/iliasm-a11y/airllm-next/pull/1).
Le snapshot de phase 1 est conservé séparément pour les mesures appariées.
La PR cible `research/airllm-next-5-generation` ; `main` n'est pas fusionnée.
Sa référence distante initiale est `9453681973ddfc93d66c6144c8eca2d5a39a8f1c`.

Sources publiées : [moteur et protocole](https://github.com/iliasm-a11y/airllm-next/commit/1e31d551010e9c836c5a34b1f14b45df19cabdf9)
et [correction des streams](https://github.com/iliasm-a11y/airllm-next/commit/b582fe96337982528f213a18b0e4fe00ddab1fe6).
Les arbres Git sont vérifiés identiques aux checkpoints locaux `5a7ea34` et
`51be52d` ; les métadonnées de commit du connecteur peuvent différer.

Les anciens résultats restent dans `IO_WALL_RESEARCH_FR.md` et
`evidence-io-wall/`. Les nouveaux journaux sont dans `evidence-io-phase2/`.
Les SHA-256 des sources au démarrage des processus et des checkpoints
identifient le code réellement mesuré ; des mesures préliminaires peuvent
donc porter sur des sources antérieures à la correction finale.

## Lectures et cache système

Le rapport détaillé est [PHYSICAL_IO_PHASE2_FR.md](PHYSICAL_IO_PHASE2_FR.md).

- `pread_nocache` demande `F_NOCACHE` sur une ouverture séparée ; cette politique
  ne fuit pas vers les descripteurs normaux. Une plateforme incompatible ou un
  échec système provoque une erreur explicite. Le lecteur compressé refuse cette
  option tant que son transport ne l'implémente pas.
- Les petites régions macOS, jusqu'à 8 Mio, utilisent directement les octets
  renvoyés par `pread` lorsque la lecture est complète. Cela évite le
  préremplissage d'un `bytearray` ; sur le repli `pread`, une copie supplémentaire
  est aussi évitée. Les lectures partielles, interruptions,
  erreurs, plafonds et emprunts concurrents de descripteurs restent contrôlés.
- `physical_io.py` fournit le compteur processus macOS
  `proc_pid_rusage(RUSAGE_INFO_V2).ri_diskio_bytesread`. Son ABI installée a été
  vérifiée par compilation. Linux conserve `/proc/self/io`. Une indisponibilité
  reste `null` ; les octets logiques ne remplacent jamais ce compteur.
- Le banc conserve séparément les compteurs globaux du périphérique, qui
  incluent aussi l'activité étrangère. Aucun de ces compteurs ne mesure les
  cellules NAND ou un débit propre au fichier au niveau matériel.

Le banc crée un safetensors pseudo-aléatoire de 128 Mio, utilise des sélections
de 1 Mio dans un ordre mélangé identique, impose les hachages exacts et supprime
son fichier temporaire après exécution. Cinq répétitions, processus frais et
ordres mélangés comparent les deux révisions. Le temps de hachage est séparé du
temps de lecture. Les compteurs mesurés excluent l'échauffement déclaré.

| Cas, 128 Mio | Phase 1 | Phase 2 | Interprétation |
|---|---:|---:|---|
| `pread`, échauffement explicite | 7,078 ms | 6,354 ms | 1,114× sur le chemin de lecture ; compteur stockage processus nul |
| `pread`, cache non contrôlé | 7,580 ms | 7,021 ms | 1,080× ; aucune qualification de SSD froid |
| `mmap`, échauffement explicite | 12,792 ms | 12,764 ms | Écart négligeable |
| Fichier créé nocache, lecture nocache | 54,219 ms | 53,216 ms | 1,019× ; gain trop faible pour une accélération physique robuste |

Les **70 processus** rendent les hachages attendus. La série avec création
nocache observe **131 596 288 à 135 532 544 octets** de stockage par processus,
avec progression proche de `disk0`. À l'inverse, toutes les lectures de la
première série ont un delta processus nul, y compris certaines lectures
nocache. **`F_NOCACHE` ne purge pas le cache OS et ne suffit pas à qualifier une
mesure de physique.** Le RSS maximal du banc reste inférieur à 38 Mio.

Preuves : [60 essais comparés](evidence-io-phase2/physical-paired-128m/report.json),
[10 essais avec activité de stockage](evidence-io-phase2/physical-nocache-created-128m/report.json),
[ABI du compteur](evidence-io-phase2/physical-counter-abi.json).

## Course CPU/GPU découverte sur le vrai modèle

La première comparaison Granite sur Metal respectait les tolérances officielles,
les tokens et les routages, mais échouait à l'égalité bit à bit entre options de
prélecture. Le contrôle sans prélecture était répétable. L'échec a été conservé.

L'audit a identifié une vraie course : dans MLX 0.32.4, le périphérique par défaut
est global au processus. Les contextes `mx.stream(mx.cpu)` des workers pouvaient
donc changer le placement des opérations du thread principal pendant la
construction de son graphe GPU. Les trois chemins F32/BF16/FP8 utilisent désormais
des opérations de décodage explicitement placées sur CPU, sans modifier ce défaut
global. Aucun verrou n'est conservé autour d'une attente de future.

Un test avec barrière et vraie lecture SingleFlight fait échouer les **trois
chemins de l'ancien lecteur** ; les trois passent avec le correctif. La nouvelle
série sur le checkpoint entraîné complet devient **bit à bit identique entre
variantes Metal**, et identique au témoin initial sans prélecture. Aucun seuil
numérique n'a changé. [Preuve de concurrence](evidence-io-phase2/worker-stream-race.json),
[rapport du modèle](GRANITE_MOE_PHASE2_FR.md).

Les mesures antérieures à ce correctif restent historiques. Elles ne constituent
pas une certification du placement exclusif des opérations sur GPU.

## SpecPrefetch : causes, corrections et résultat négatif restant

Le [rapport ciblé](SPEC_PREFETCH_PHASE2_FR.md) conserve les profils, essais
intermédiaires et contrôles. Trois coûts distincts ont été isolés :

1. La recherche parcourait toutes les arêtes de transition, y compris celles
   des autres couches. L'index est maintenant borné par couche et expert source.
2. Observer un routage réel pouvait attendre une mauvaise prélecture active.
   Son retrait est maintenant finalisé par callback ; ses deux réservations
   restent détenues jusqu'à la fin réelle. `drain` et `close` attendent toujours.
   Les callbacks utilisent des références faibles pour ne pas conserver les
   payloads après libération des réservations. Les échecs optionnels restent
   ignorés, puis la demande exacte réessaie normalement.
3. Sur le petit modèle, les premières prélectures lançaient un second worker,
   associé à une hausse durable des temps de chargement et d'attente.
   Un contrôle avec prédiction/observation mais zéro prélecture retrouve la
   latence sans option ; les compteurs confirment l'absence de lectures ajoutées
   à chaud. L'alternance effective des tâches entre threads reste une hypothèse,
   pas une trace observée. Un seuil de précision ou un échauffement seul ne suffisait pas.

La correction combine **quatre prédictions observées sans lancer d'E/S**, un
historique glissant borné de précision et un coût de planification estimé sur
les vraies demandes exactes. L'admission compare ce coût et les coûts de lecture
observés ; aucune vitesse SSD supposée ni seuil de taille arbitraire n'est
introduit. Il s'agit d'un contrôle heuristique de trafic, pas d'une prévision
certifiée de gain. Le chemin SpecPrefetch désactivé n'ajoute pas ces chronomètres.

La comparaison finale mélange cinq rondes de processus frais, deux prompts,
trois répétitions chaudes et 16 tokens, sous les mêmes 32 Kio de cache logiciel :
**30 requêtes chaudes par variante** sur le modèle GLM miniature aléatoire.

| Variante | Médiane chaude | Octets logiques/requête |
|---|---:|---:|
| Phase 1, SpecPrefetch actif | 74,403 ms | 4 158 784 |
| Phase 2, SpecPrefetch actif | 61,740 ms | 4 094 272 |
| Phase 2, SpecPrefetch désactivé | 59,319 ms | 4 094 272 |

La correction réduit ici la durée de l'option d'environ **17,0 %**, soit **1,205×**
par rapport à son ancienne version. Elle reste **environ 4,1 % plus lente que
l'option désactivée**. SpecPrefetch demeure désactivé par défaut. Tous les logits
enregistrés et les tokens sont identiques bit à bit ; les hachages natifs sont
stables avant/après chaque processus et entre processus de la série finale.

[Mesures finales après correction des streams](evidence-io-phase2/specprefetch-after-stream-fix.json).
La série précédente, cohérente au stade d'admission seul,
`specprefetch-paired-stable.json`, mesurait 75,445 → 60,950 ms, contre 58,014 ms
sans l'option. Elle reste disponible comme étape intermédiaire.
La série précédente `specprefetch-final.json` est un essai préliminaire : elle
précède cette correction et son indicateur de stabilité est faux à cause d'un
ajout d'export de module pendant l'expérience. Elle est conservée et n'est pas
utilisée comme résultat final.

Le contrôle de métadonnées à 4 096 arêtes mesure séparément environ
951,4 → 11,6 ms pour 8 192 appels. Ce facteur concerne uniquement le prédicteur,
sur un essai de contrôle ; il n'est ni une accélération de génération ni une
mesure de SSD. Les relectures de routes du banc utilisent de vrais fichiers
locaux, sans délai injecté, avec cache OS non contrôlé et résultats négatifs
conservés.

## Régression contrôlée SmolLM2 sur Metal

Checkpoint entraîné complet `HuggingFaceTB/SmolLM2-135M-Instruct`, même fichier
de poids que la phase 1 : SHA-256
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
FP32, 16 tokens, deux prompts, trois ordres de processus, deux répétitions
chaudes : **12 requêtes chaudes par variante**. Les budgets restent 256 Mio de
poids, 32 Mio d'E/S, 32 Mio de KV, 128 Mio de workspace et 1 Gio de réserve.

Série finale après correction des streams :

| Variante | Médiane chaude | Octets logiques/requête | Pic RSS processus |
|---|---:|---:|---:|
| Phase 1, adaptatif + pipeline | 0,594 s | 2 195 940 096 | 700,2 Mo |
| Phase 2, options nouvelles désactivées | 0,867 s | 4 304 506 752 | 683,0 Mo |
| Phase 2, adaptatif + pipeline | 0,595 s | 2 212 751 232 | 700,2 Mo |

**Tous les logits enregistrés et les tokens gloutons sont identiques bit à bit**
entre ces variantes. Les sources natives sont stables entre les processus de
cette série. Les coûts observés alimentant le cache adaptatif peuvent modifier
l'admission et donc les octets lus, même avec les mêmes tokens et budgets.

Cette série ne démontre **pas de gain de génération de la phase 2 sur la
phase 1** : l'écart médian adaptatif/pipeline est ici d'environ 0,3 %, négligeable
à cette échelle. Le gain par rapport à l'option désactivée appartient surtout aux
mécanismes de cache/pipeline déjà présents en phase 1. L'amélioration du petit
chemin `pread` ne garantit pas celle d'une génération complète.

La série préliminaire `smollm-metal.json`, antérieure à la correction des
streams, reste conservée : 0,685 → 0,731 s pour adaptatif/pipeline, environ
6,7 % plus lent ; nocache seul 0,907 s et adaptatif/pipeline/nocache 0,697 s.
Tous les logits de cette série étaient aussi identiques entre variantes.

Les régimes du cache OS varient : une ronde adaptatif/nocache observe environ
1,66 à 1,90 Go selon la requête, tandis que les deux autres affichent zéro octet
physique. Les autres variantes ont leurs propres deltas dans
les journaux. Les médianes mélangées ne servent donc à aucune comparaison de
débit SSD. L'hôte exécute d'autres travaux Python ; il n'est pas globalement
isolé et du swap système préexistait.

[Comparaison finale](evidence-io-phase2/smollm-metal-after-stream-fix.json),
[mesures finales par processus](evidence-io-phase2/smollm-metal-after-stream-fix-runs/),
[ancienne série et essais nocache](evidence-io-phase2/smollm-metal.json).

## Véritable MoE entraîné intermédiaire

Le checkpoint complet **IBM Granite 3.0 1B-A400M Instruct**, environ 1,3 milliard
de paramètres et 400 millions actifs, a été téléchargé et vérifié : 24 couches,
32 experts par couche, top-k 8 ; poids BF16 de **2 669 283 096 octets**.
Révision `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`, SHA-256
`f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc`.
Les configurations et tokenizer sont aussi vérifiés contre leurs blobs Git.

Le nouveau backend explicite `AirLLMNextGraniteMoeModel` lit les seuls experts
sélectionnés et partage les budgets Next. Il ne change pas le dispatch de
production. La référence utilise les équations officielles Transformers,
chargées couche par couche en FP32 à partir des mêmes poids BF16, avec
libération sur `meta` ; elle n'utilise ni mathématiques ni lecteur AirLLM.

Sur **deux prompts de 15/18 tokens et quatre nouveaux tokens par prompt**, les
contrôles CPU puis Metal corrigé donnent les mêmes tokens gloutons et les mêmes
ensembles d'experts à chaque couche/étape. Les logits respectent les seuils
prédéfinis `atol=1e-4`, `rtol=5e-4` face à la référence officielle ; les variantes
natives de chaque device sont **identiques bit à bit**. L'erreur maximale
officielle Metal corrigée est **8,1062×10⁻⁵**.

| Metal après correction, quatre tokens | Médiane des deux prompts |
|---|---:|
| Natif sans prélecture | 1,306 s |
| Prélecture exacte | 1,071 s |
| Prélecture exacte + SpecPrefetch | 1,041 s |

Ces deux requêtes courtes ne démontrent pas de gain général. Les compteurs de
stockage et le cache OS varient entre processus ; la référence indépendante
emploie en outre une autre stratégie de chargement. Aucun rapport des durées
référence/natif n'est présenté comme une accélération de noyau.

La répétition CPU après correction reste bit à bit identique aux tableaux de la
série CPU initiale, y compris la référence. Le RSS final observé reste sous
**777 Mo sur CPU et 752 Mo sur Metal**, avec
**256 Mio de cache poids**, **32 Mio d'E/S**, **32 Mio de KV** et **128 Mio de
workspace**. Le KV réellement relevé est 12 Mio ; les E/S en vol restent dans
leur plafond. Ces payloads ne sont pas un plafond du RSS global. Le modèle
complet tient dans la RAM physique : ce résultat valide l'exécution sélective
d'un vrai MoE intermédiaire, **pas un modèle dépassant la RAM**.

[Rapport détaillé, avant/après et reproduction](GRANITE_MOE_PHASE2_FR.md),
[résultat strict Metal](evidence-io-phase2/granite/granite-metal-after-stream-fix.json),
[résultat strict CPU corrigé](evidence-io-phase2/granite/granite-cpu-after-stream-fix.json).

## Tests et reproduction

La référence a été exécutée à nouveau : **720 tests réussis, 1 ignoré,
11 sous-tests réussis**, en 14,24 s. [Journal de référence](evidence-io-phase2/baseline-tests.log).
La suite finale donne **787 tests réussis, 1 ignoré, 11 sous-tests réussis** en
15,12 s, sans assouplissement de tolérance numérique :
[journal complet](evidence-io-phase2/tests-final.log).
Les options numériques des tests unitaires utilisent leur configuration CPU ;
les générations entraînées Metal sont mesurées séparément.

La première exécution générale a révélé deux tests attachés au syscall `preadv`
et à une représentation particulière de tampon `bytearray`. Ils vérifient
désormais le syscall réellement utilisé, y compris sans `preadv`, et le
propriétaire partagé des tampons, leur taille et leurs octets exacts. Les
assertions d'interruption, lecture courte et fin de fichier restent présentes.
Le [premier journal avec échecs](evidence-io-phase2/tests-first-run.log) est conservé.

La wheel a été construite et installée hors checkout : **67 sources Python
identiques**, nouveaux modules et export Granite présents, et six tokens
SmolLM2 Metal identiques au résultat attendu. La version reste `5.0.0a1`, sans
publication PyPI. [Vérification du paquet](evidence-io-phase2/package-verification.json).

Le script de validation finale est `scripts/validate_io_phase2.sh` ; il travaille
hors ligne et n'initie aucun téléchargement de poids.

Les JSON, journaux et compteurs sont versionnés. Les grands tableaux `.npz`,
profils binaires, XML, fixtures synthétiques et wheels restent dans la livraison
locale ; [leur manifeste](evidence-io-phase2/local-artifacts-manifest.json) donne
leurs chemins canoniques, tailles et SHA-256. Ils ne sont pas distribués comme
des poids entraînés. Les benchmarks peuvent régénérer les tableaux ; le
téléchargeur explicite vérifie le checkpoint officiel avant utilisation.

```sh
AIRLLM_PYTHON=/chemin/python bash scripts/validate_io_phase2.sh

PYTHONPATH=air_llm python benchmarks/benchmark_physical_io.py \
  --output evidence-io-phase2/nouvelle-serie-physique \
  --baseline-source /chemin/snapshot-3e4138a \
  --size-mib 128 --chunk-kib 1024 --repeats 5 \
  --modes pread_nocache --fixture-write-policy nocache \
  --scenarios uncontrolled_after_creation_and_prior_jobs
```

Pour une future mesure du Fanxiang, ajouter `--fixture-parent` avec son vrai
point de montage puis vérifier la chaîne de périphériques du rapport. Le chemin
`/Volumes/SSD` observé ici est un lien vers `/`, donc vers le stockage interne.

## Limites

Les budgets de payload du cache, des futures et du KV ne sont pas le RSS total.
Les processus, allocations MLX, buffers de référence, métadonnées et compteurs
système sont rapportés selon leur portée propre. Le swap système et les
compteurs du périphérique incluent les autres applications.

Restent hors validation : Fanxiang physique, autres contrôleurs/boîtiers,
GLM/Qwen géants, quantification Granite, longs contextes, qualité générale du
modèle, performance thermique prolongée, énergie et temps GPU purs. Les options
de recherche restent explicites et aucune fusion de `main` n'est autorisée.
