# Préchargement dense d'une matrice à l'avance — prototype désactivé

Le patch `patches/llama-dense-lookahead.patch` ajoute une seule hypothèse : conseiller la lecture du prochain poids dense CPU pendant l'exécution du nœud courant. Il ne crée aucune copie des poids et ne modifie ni les octets GGUF, ni la quantification temporaire, ni les kernels mathématiques ou leur ordre. Aucune opération n'est ajoutée au graphe. Le layout public de `ggml_cplan` reste inchangé.

Options : `AIRLLM_DENSE_LOOKAHEAD=1`, et `AIRLLM_DENSE_LOOKAHEAD_MIB=16|32|64|128`. La fenêtre par défaut est **32 MiB**, le plafond absolu **128 MiB**. Une valeur inconnue désactive le préchargement. Les 600 agents logiques, le slot physique unique et les garde-fous externes restent inchangés.

Le wrapper du backend CPU prépare un plan borné de 1 024 entrées au maximum, à partir du graphe réellement soumis. Il retient les MUL_MAT mono-colonnes, dont le poids est contigu, constant et associé au vrai buffer `CPU_Mapped` d'usage `WEIGHTS`. Les adresses doivent appartenir au buffer et toutes les pages inspectées être lisibles et non modifiables. Les activations, résultats, allocations propriétaires modifiables et buffers Metal sont exclus. Le préchargement est limité à un préfixe de la prochaine matrice et aux pages de bord que le calcul devra lui-même lire.

Un unique worker et une unique tâche en vol sont admis dans le processus. Un graphe concurrent désactive sa propre anticipation. Le thread CPU zéro signale l'index réellement atteint avant le calcul, sans changer les barrières ou les calculs. Aucun futur poids n'est supposé disponible : si le worker est encore occupé, la nouvelle anticipation est omise. L'annulation supprime une tâche en attente ; un appel système déjà commencé est drainé avant le retour du calcul et avant la libération du graphe ou du modèle. L'enregistrement global reste réservé pendant ce drainage.

Chaque tâche exige une pression macOS normale, au moins **2 GiB + taille de fenêtre** de pages libres/inactives estimées, et une croissance de mémoire physique du compresseur **inférieure à 128 MiB depuis le début du graphe**. La quantité de compresseur déjà occupée par les autres applications n'est pas assimilée à une nouvelle allocation du prototype. Une mesure indisponible refuse la tâche. Ces estimations ne remplacent pas les limites externes RSS 8,5 GiB, swap 32 MiB et pression mémoire.

Un résumé `AIRLLM_DENSE_LOOKAHEAD` enregistre entrées, tâches, appels, octets réellement conseillés, résultats/erreurs, temps passé dans l'appel, attentes de fermeture, omissions, mémoire disponible minimale, pression et croissance du compresseur. Un retour zéro de `posix_madvise(POSIX_MADV_WILLNEED)` signifie seulement que le conseil a été accepté. Ce n'est ni la preuve d'une lecture physique, ni un compteur de cache hit, ni une mesure de chevauchement utile.

La portée du plan est celle du segment CPU soumis au backend. Le profilage qui demande un callback à chaque opération peut le fragmenter en graphes ne contenant qu'un seul poids : aucune anticipation utile n'est alors programmée. Les segments CPU d'un chemin hybride peuvent l'utiliser quand ils contiennent au moins deux poids éligibles ; les poids Metal restent exclus. Il faut comparer les compteurs et le débit d'un graphe non sérialisé, puis vérifier séparément toutes les familles de fidélité sur le même runtime natif.

## Fixture prévue

`dense_lookahead_fixture.cpp` crée de vrais buffers GGML `CPU_Mapped` à partir d'un petit fichier temporaire en lecture seule. Elle couvre alignement et bornes, fenêtre plafonnée, refus de mémoire modifiable, flags d'alias à la frontière de profondeur, absence de concurrence, refus d'admission, échec d'appel injecté et drainage d'une tâche volontairement bloquée. Un appel de conseil réel est également effectué, puis un petit graphe CPU est comparé avec et sans l'option. Les hooks d'injection servent uniquement à rendre les états de cycle de vie déterministes ; ils ne suppriment pas la validation des poids et des plages.

```sh
AIRLLM_TEST_DENSE_LOOKAHEAD=1 \
AIRLLM_PROFILE_RUNTIME_SOURCE=/absolute/path/llama-v3 \
AIRLLM_PROFILE_RUNTIME_LIBRARIES=/absolute/path/runtime-v3-build/bin \
python -m pytest -q research/glm53_gpu_v3/test_dense_lookahead.py
```

La fixture a ensuite été compilée et exécutée contre les bibliothèques v3 : **2 tests réussis, 26 contrôles GGML**. Le [rapport brut](evidence/dense-lookahead-fixture.json) conserve stdout/stderr et les hashes du helper, de la fixture et des bibliothèques. L'appel de conseil réel de 16 KiB retourne zéro. Le petit graphe natif avec l'option active enregistre lui aussi un appel du worker, sous pression normale, sans croissance du compresseur dans ce graphe, puis les mêmes sorties CPU en bits. Ces faits montrent l'admission et l'exécution du mécanisme ; ils ne prouvent pas des lectures physiques ni un chevauchement bénéfique.

La première tentative de fixture a déclenché l'assertion d'alignement du buffer GGML sur son entrée FP32 de pile. La fixture a été corrigée avec `alignas(64)` ; aucun code ou binaire du runtime n'a été modifié pour cette correction.

Le [préflight de série](evidence/patch-series-validation.json) applique les 15 patches dans un clone isolé et vérifie les 11 fichiers natifs résultants, dont le nouveau header, contre `llama-v3`. Le layout mathématique et les sources produites sont identiques. Les changements CPU de ce prototype n'ont pas modifié `llama-v2` ni `llama-mmap.cpp`. Aucun gain du modèle, disparition de copie ou comportement de cache physique n'est qualifié par ces preuves de composant et de source.
