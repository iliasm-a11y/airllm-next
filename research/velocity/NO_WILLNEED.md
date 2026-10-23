# Préchargement mmap : hypothèse vérifiée dans les sources, essai négatif

Le binaire Ollama installé annonce llama.cpp `0.7.0-dev`, commit `620d2225c`.
Le commit primaire correspondant est
[`620d2225cc436ab0b001e3200e590e3b286ef9ab`](https://github.com/ggml-org/llama.cpp/commit/620d2225cc436ab0b001e3200e590e3b286ef9ab).
Cette correspondance de version ne prouve pas l'absence de patches Ollama.

Les sources contiennent un préchargement potentiellement défavorable à une
machine de 16 Gio :

1. [`llama-model.cpp`, ligne 1876](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/src/llama-model.cpp#L1876)
   appelle `init_mappings(true, ...)`.
2. [`llama-model-loader.cpp`, ligne 1430](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/src/llama-model-loader.cpp#L1430)
   choisit `SIZE_MAX` comme taille de préchargement lorsque mmap est actif.
3. [`llama-mmap.cpp`, lignes 513–515](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/src/llama-mmap.cpp#L513)
   conseille `POSIX_MADV_WILLNEED` pour tout le fichier sauf les plages lazy.

`--no-warmup` ne désactive pas ce chemin de chargement. `--lazy-mode on` ne
rend pas automatiquement tous les experts lazy. Aucune option CLI de
désactivation de ce préchargement n'a été identifiée dans le code inspecté.
L'hypothèse testée est de supprimer uniquement ce conseil d'anticipation, en
conservant les lectures à la demande et tous les calculs du modèle.

## Prototype et vérification locale

`no_willneed.c` fournit une interposition macOS limitée au processus choisi.
Elle retourne zéro pour `MADV_WILLNEED`; les autres conseils sont transmis à
libc. Elle ne modifie ni les données des tenseurs, ni mmap, ni les protections
mémoire. Le probe local confirme que les chemins `posix_madvise` et `madvise`
atteignent cette interposition sur ce Mac.

Les sorties du probe normal et du probe interposé sont identiques pour les
deux conseils WILLNEED valides, pour `POSIX_MADV_RANDOM`, et pour les deux
appels à conseil invalide : retour `-1`, errno `EINVAL`. Le journal interposé
compte deux suppressions de 16 384 octets. Une relecture indépendante a
contrôlé le code et ces résultats.

Limite intentionnelle : WILLNEED retourne zéro sans vérifier adresse et
longueur. Ce mécanisme n'est donc pas une interposition universellement
transparente, même s'il ne change pas les équations du modèle dans l'usage
visé. Une première variante de développement utilisant `dlsym(RTLD_NEXT)`
a bloqué le petit probe et a été remplacée avant tout essai sur modèle.

Le binaire original est conservé. Une extraction arm64 privée est signée
ad hoc dans le répertoire de travail pour permettre l'injection explicite;
ses dépendances arm64 vérifiées sont uniquement des bibliothèques/frameworks
système. `--version` affiche le marqueur `AIRLLM_NO_WILLNEED_READY`, ce qui
prouve le chargement de la bibliothèque dans cette copie. Le script
`build_no_willneed.sh` reproduit compilation, probe, copie et signature.

## Résultat sur le modèle réel

Le journal `evidence-velocity/llamacpp-installed-cpu-no-willneed-smoke.log`
contient le marqueur READY mais **aucune suppression WILLNEED observée**.
Le garde-fou a arrêté le processus pendant son initialisation, avant qu'il
serve une requête de génération. Aucun token n'a été obtenu dans cet essai.
Le garde-fou suit le swap système global; il ne permet pas à lui seul
d'attribuer chaque octet de swap au processus surveillé.

Le chemin de préchargement existe dans les sources, mais son rôle causal dans
cette interruption n'est pas démontré. L'absence d'appels interceptés peut
correspondre à un arrêt antérieur ou à un chemin différent dans le binaire.
Une trace supplémentaire serait nécessaire pour trancher. Ce prototype ne
démontre donc ni correction du chargement réel, ni accélération de génération,
ni validation sûre du niveau A. Aucun autre essai massif n'a été lancé après
ce résultat négatif.
