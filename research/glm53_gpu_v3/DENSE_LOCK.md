# Rétention dense CPU bornée, sans copie des poids

Cette expérience supplémentaire est désactivée par défaut et séparée du runtime
V3 déjà mesuré. Elle appelle `mlock` sur des pages originales du GGUF, mappées en
lecture seule, pour retenir un petit ensemble fixe de matrices F32/BF16 entre
tokens. Elle ne modifie ni les poids, ni les codecs, ni le graphe, ni l'ordre des
opérations. Les matrices Q8 et les experts sont exclus afin d'éviter un second
budget sur les poids du chemin GPU sélectif. Aucun gain de décodage ne découle
de la fixture ou de la sélection présentées ici.

Le patch [llama-dense-lock.patch](patches/llama-dense-lock.patch) ajoute un
propriétaire RAII au modèle GLM5_NEXT, après le chargement des mappings et avant
la création du contexte. Son membre est déclaré en dernier : sa destruction
libère les verrous avant les buffers et les mappings du modèle. Aucun thread,
buffer de poids supplémentaire ou changement des 600 agents logiques et du
slot physique partagé n'est introduit.

## Admission et échecs

- Activation explicite : `AIRLLM_DENSE_LOCK=1`,
  `AIRLLM_DENSE_LOCK_MIB=128|256|512` (défaut 256), et
  `AIRLLM_DENSE_LOCK_NAMES`, noms exacts séparés par des virgules, au plus
  128 noms et 16 Kio. Une liste absente ou invalide ne retient rien.
- Chargement mmap original uniquement, hors `use_mlock` préexistant. Chaque
  poids doit être constant, contigu, F32/BF16, dans une famille dense autorisée,
  dans un buffer `CPU_Mapped` de type WEIGHTS. La plage complète de pages doit
  appartenir aux mappings du modèle et être lisible, non inscriptible et sans
  verrou utilisateur existant. Les aliases inconnus sont refusés.
- Les pages partagées entre poids ne sont comptées qu'une fois ; les deux pages
  de bord éventuelles comptent intégralement. Le maximum est 512 Mio de pages,
  sans augmenter `RLIMIT_MEMLOCK`. Il faut une pression système normale et une
  estimation libre + inactive d'au moins 2 Gio plus le volume restant à retenir.
  Ce calcul est une admission prudente, pas une garantie de mémoire future.
- Un seul propriétaire est admis dans le processus. Avant chaque appel,
  l'admission et les protections sont revérifiées ; annulation et contrôle final
  peuvent provoquer une libération complète. Le succès de `mlock` et la
  présence de toutes les pages via `mincore` sont vérifiés.
- Tout échec déclenche la libération des plages déjà tentées, sans déclarer un
  cache retenu. Si `munlock` échoue encore lors de la destruction, l'admission
  est définitivement fermée dans ce processus pour ne pas ajouter un autre
  budget aux pages dont la libération reste incertaine.

Les gardes du superviseur — RSS 8,5 Gio, croissance globale du swap 32 Mio,
pression et timeout — restent nécessaires et inchangées. Des pages verrouillées
peuvent accroître la pression sur les autres pages. L'absence de copie anonyme
ne signifie pas une absence de coût mémoire ou de coût au chargement.

## Plan de 256 Mio et portée des coûts

Le [plan effectivement généré](evidence/dense-lock-plan256.json) contient
**103 matrices, 266 862 592 octets de poids et 268 255 232 octets de pages**,
réunis en 85 plages. Le plafond est 268 435 456 octets. Les pages de bord
ajoutent donc 1 392 640 octets au volume des poids sélectionnés.

`select_dense.py` ne lit aucun payload de checkpoint. Il réconcilie les
684 matrices et leurs 9 444 458 496 octets avec les identités natives du pas 1
de C16 (tête GPU), puis vérifie les 684 tailles ordonnées sur chacun des trois
pas de l'ancien profil CPU4. Les temps anciens sont attachés aux identités par
ordre du graphe et tailles exactes : ce n'est pas une mesure simultanée de nom,
pointeur et durée. La sélection gloutonne maximise le temps observé par octet
de page ajouté ; elle n'est pas un optimum global.

Les opérations sélectionnées représentent 0,33921 s sur les 6,28040 s moyens
de cet ancien profil sérialisé. Même en supprimant **tout** leur temps, calcul
compris, le gain de débit conditionnel serait seulement **5,7094 %**. Leur temps
réel ne peut pas être entièrement éliminé par la rétention. Cette sélection
seule ne justifie donc pas un objectif de +20 %. La combinaison avec les
projections Q8 GPU est une hypothèse distincte à vérifier intégralement.

Le plan lie les statistiques courantes des cinq champs des shards à un
manifeste antérieurement vérifié et à ses SHA-256 originaux. Dans ce cas local,
le périphérique et un ctime ont changé ; le plan les expose. Il ne revendique
ni une nouvelle vérification intégrale des 110 Go ni l'égalité des cinq champs
historiques. Un nouveau manifeste vérifié du même checkpoint est accepté ; le
hash JSON local historique n'est pas exigé pour les autres utilisateurs.

## Vérifications effectuées et reproduction

La [fixture réelle](evidence/dense-lock-fixture.json), sans modèle ni GPU, a
réussi 38 contrôles normaux et 10 contrôles dans un processus séparé de
défaillance de libération : vrai `CPU_Mapped` en lecture seule, `mlock`,
`mincore`, libération, union des pages, budget incluant les bords, admission,
annulation, rollback partiel, exception injectée, concurrence, ordre de durée
de vie et limite système inchangée. Le patch passe `git apply --check` contre
les sources V3 sans les modifier. Cela ne remplace pas compilation du runtime,
fidélité complète et mesures ABBA de la nouvelle bibliothèque.

Depuis la racine du dépôt, avec les bibliothèques et sources épinglées déjà
construites, la petite fixture se lance explicitement :

```sh
AIRLLM_TEST_DENSE_LOCK=1 \
AIRLLM_PROFILE_RUNTIME_SOURCE=/path/to/pinned/source \
AIRLLM_PROFILE_RUNTIME_LIBRARIES=/path/to/runtime/bin \
python3 -m pytest -q research/glm53_gpu_v3/test_dense_lock.py
```

Elle réserve le même verrou de benchmark ; elle est ignorée si ce slot est
occupé. Elle ne relève jamais la limite memlock. Une nouvelle sélection requiert
les trois fichiers de métadonnées et le manifeste de vérification :

```sh
python3 research/glm53_gpu_v3/select_dense.py \
  --inventory research/glm53_m5/sources/gguf-headers.json \
  --source-plan /path/to/C16/capture/source_plan.jsonl \
  --operations /path/to/cpu-pages4/operations.jsonl \
  --model-directory /path/to/GLM-5.3-Flash-Q2_K_S \
  --identity-manifest /path/to/model-verification.json \
  --budget-mib 256 --output /path/to/new-plan256.json
```

`--expected-identity-manifest-sha256` permet de fixer aussi le hash du JSON de
vérification. Le plan enregistre toujours son hash réel. Le constructeur isolé
inclut le seizième patch uniquement avec `--dense-lock-experiment` ; l'exécution
reste désactivée par défaut. Les runners utilisent `--dense-lock-plan` pour
vérifier les identités, figer les trois variables et exiger le journal natif de
rétention conforme. La référence A doit avoir ces trois variables absentes.
