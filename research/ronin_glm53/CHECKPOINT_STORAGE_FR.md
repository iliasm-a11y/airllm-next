# Audit du checkpoint et protocole de téléchargement

Le checkpoint GGUF est épinglé à la révision Hugging Face `119034bc3d9d76eb41f91508e6f9fb19fde42672`. Les identités LFS officielles et les tailles exactes figurent dans `checkpoint-manifest.json`. Aucun checkpoint BF16 ou FP8 complet n'a été téléchargé.

## Fichiers indispensables au chemin texte

| Fichier | Octets | Rôle |
|---|---:|---|
| GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf | 9 430 112 | Métadonnées et tokenizer ; zéro tenseur |
| GLM-5.3-Flash-Q2_K_S-00002-of-00002.gguf | 110 017 922 752 | 1 383 tenseurs |
| Total | 110 027 352 864 | 110,027 Go décimaux ; 102,471 Gio |

Le tokenizer, son vocabulaire de 154 880 entrées, ses 321 649 merges, ses identifiants spéciaux et le chat template sont intégrés au premier fichier. Aucun téléchargement supplémentaire de tokenizer n'est requis pour cette route GGUF. Les fichiers `mmproj` et `mtp` sont exclus du téléchargement texte. Cette exclusion ne constitue aucune prise en charge de la vision ou du MTP.

Le premier fichier a été récupéré intégralement puis vérifié contre son SHA256 officiel. Le second a seulement fait l'objet d'une lecture HTTP bornée à 1 Mio afin d'inspecter son en-tête ; l'index se termine à l'octet 86 719 et les données commencent à 86 720. Cette lecture contient donc aussi environ 0,92 Mio du premier tenseur ; aucun fichier complet de poids n'a été téléchargé. L'inventaire exact, avec dimensions GGUF et offsets relatifs au début des données, est dans `checkpoint-tensor-directory.json`.

## Formats réellement distribués

| Format | Nombre de tenseurs | Octets de données |
|---|---:|---:|
| Q2_K | 126 | 99 883 155 456 |
| Q8_0 | 306 | 8 953 004 032 |
| BF16 | 225 | 796 917 760 |
| F32 | 726 | 384 756 984 |

Les projections KDA `ssm_beta`, `ssm_f_a`, `ssm_f_b`, `ssm_g_a` et `ssm_g_b` sont notamment en BF16. Il serait incorrect d'annoncer la compatibilité complète de ce GGUF en n'implémentant que Q2_K et Q8_0.

Les 42 blocs MoE contiennent chacun trois tenseurs routés Q2_K de 792 723 456 octets. Un expert utilise 8 257 536 octets pour ses trois matrices. Sans cache d'experts, les huit experts choisis dans les 42 blocs représentent 2 774 532 096 octets de poids routés par token décodé. Il s'agit d'un calcul d'inventaire, pas d'une mesure des lectures SSD, ni d'une promesse de débit. Les autres tenseurs représentent environ 10,135 Go et devront eux aussi être répartis entre résidence et streaming.

## Décision de stockage

Avec l'ancien diagnostic indicatif de 125,45 Go décimaux libres, les deux shards laisseraient 15,423 Go. En réservant 64 Mio aux métadonnées et fichiers auxiliaires, la marge calculée est de 15 355 538 272 octets, soit environ 14,30 Gio. Ce scénario est refusé par la politique conservatrice de 32 Gio libres après installation. Il ne décrit pas l'état actuel du Mac : aucun volume du Mac M5 n'est visible depuis cet environnement Linux.

Le contrôle consulte le système de fichiers de la destination réelle, même quand le répertoire final n'existe pas encore. Sur macOS, il contrôle en plus le volume système `/` et refuse le téléchargement si ce volume a moins de 32 Gio libres, y compris lorsque la destination est un SSD externe. Cette réserve est une politique d'exploitation et ne garantit pas, à elle seule, l'absence de swap ou de pression mémoire pendant l'inférence.

## Commandes préparées

Précontrôle sans téléchargement ni création de répertoire :

```bash
python scripts/download_glm53_flash.py --destination /Volumes/Models/glm53-flash
```

Après validation du moteur et seulement si les contrôles passent :

```bash
python scripts/download_glm53_flash.py --destination /Volumes/Models/glm53-flash --download
```

Le script ne valide pas le moteur d'inférence : son champ `engine_validated` reste explicitement faux. Son succès signifie uniquement que les deux fichiers exacts sont disponibles et vérifiés.

## Garanties et limites du protocole

- Pas de cache Hugging Face ni de deuxième copie du checkpoint.
- Écriture séquentielle dans un `.partial` situé dans la destination définitive ; renommage local après vérification SHA256.
- Réponses HTTP `206`, bornes `Content-Range`, taille totale et taille de chaque plage vérifiées ; tampon maximal de 4 Mio plus un octet pour détecter un excès.
- Identité et SHA256 du préfixe persistés après `fsync` ; reprise uniquement après vérification de ce préfixe. Une fin de plage non validée après interruption est supprimée uniquement dans le fichier temporaire identifié du script.
- Refus de réutiliser un fichier complet corrompu, un fichier partiel non identifié, un lien symbolique ou un fichier partiel possédant plusieurs liens physiques.
- Verrou d'exclusion pour les invocations coopérantes ; un verrou ancien doit être inspecté, jamais supprimé automatiquement.
- Réévaluation de l'espace libre au départ et avant chaque plage ; arrêt si la réserve est menacée.
- Les fichiers personnels, checkpoints existants et fichiers partiels dont l'identité est incertaine sont préservés.

Les tests du protocole utilisent de petits payloads indépendants du modèle : ils prouvent les contrôles d'identité, d'interruption et de stockage, pas une inférence GLM. Résultat enregistré : 22 tests réussis. Le précontrôle effectivement exécuté sur Linux et son refus du téléchargement sont conservés dans `download-preflight-linux.json`.
