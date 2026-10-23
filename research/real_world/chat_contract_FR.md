# Contrat conversationnel Qwen3.6 : vérification réelle

Audit du 23 octobre 2026, sans charger les poids et sans inférence GPU. Fichier
testé : GGUF installé de 23 938 321 664 octets, architecture `qwen35moe`.
Le lecteur a consulté seulement son index et les métadonnées : **0 octet de
tenseur lu**. Résultats détaillés : [chat_contract.json](chat_contract.json).

Le vocabulaire officiel local et les **247 587 merges BPE** correspondent
exactement aux entrées GGUF. La référence de tokenisation est le moteur chargé
directement depuis le `tokenizer.json` officiel, sans reconstruction. Le rendu
conversationnel de référence applique le template Jinja du
`tokenizer_config.json` officiel via Transformers 5.20.2. Les deux fichiers
étaient déjà présents dans le travail de la mission précédente ; aucun poids
supplémentaire n'a été téléchargé.

| Vérification | Résultat |
|---|---:|
| Textes : français, Unicode décomposé, chinois, arabe, hindi, emoji, chiffres, espaces, balises | 9/9 encodages identiques au JSON officiel |
| Décodage avec et sans suppression des tokens spéciaux | 18/18 identiques |
| Conversations × pensée activée/désactivée × conservation pensée × ajout préfixe | 64/64 rendus et suites d'IDs identiques |
| Historiques invalides : vide, aucun utilisateur, système après le début, réponse outil sans question | 5/5 refusés par AirLLM et la référence |
| Ancienne version `ac93b5d`, configurations dans son périmètre | 20/32 rendus différents ou refusés |

Les scénarios incluent système + utilisateur, plusieurs tours, espaces de bord,
assistant déjà terminé, raisonnement dans `content`, raisonnement dans
`reasoning_content` et contenus vides. Cette matrice prouve le contrat de rendu
sur ces cas ; elle ne prouve pas la qualité de génération.

## Défauts corrigés

- `enable_thinking=True` omettait le préfixe `<think>` attendu.
- Un assistant après la dernière question n'avait pas l'enveloppe de pensée
  attendue, même lorsque le raisonnement était vide.
- Les anciens raisonnements étaient refusés ; le template officiel les extrait,
  les omet par défaut lorsqu'ils précèdent la dernière question, ou les conserve
  avec `preserve_thinking=True`.
- L'adaptateur acceptait un système inséré au milieu de l'historique et des
  historiques sans question utilisateur, que le template officiel refuse.

Le cas précédent « un utilisateur, pensée désactivée » conserve exactement le
même rendu. Les espaces de bord restent rognés conformément au template
officiel. L'historique brut peut donc différer de l'historique rendu. Une session
qui réutilise son cache doit vérifier les IDs complets du préfixe et reconstruire
le cache en cas de divergence ; conserver la pensée facilite cette continuité
sans la garantir pour tous les espaces ou découpages BPE.

Les cinq IDs contrôlés correspondent aux assets officiels : `im_start=248045`,
`im_end=248046`, `endoftext=248044`, `think=248068`, `/think=248069`.
Le tokenizer expose `im_end` comme EOS. La
[configuration de génération officielle](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/generation_config.json)
prévoit les deux arrêts `248046` et `248044`, et indique `do_sample=true`,
`temperature=1.0`, `top_k=20`, `top_p=0.95`. Les paramètres effectivement employés
par la campagne doivent être enregistrés séparément ; un décodage glouton est
un autre choix de génération.

## Différence du comparateur Transformers

Une divergence hindi initiale a été expliquée sans modifier le moteur AirLLM.
Sur ce runtime, `AutoTokenizer.from_pretrained(...)` choisit `Qwen2Tokenizer` et
reconstruit son pré-tokenizer avec une regex qui omet la catégorie Unicode
`\p{M}`. Le `tokenizer.json` officiel inclut cette catégorie. Par exemple,
`हिन्दी` donne `[215989, 150127, 177453]` avec AirLLM et le JSON officiel direct,
mais `[90703, 181077, 28640, 148651, 42201]` avec ce comparateur Transformers.
Le texte décodé est identique, les entrées du modèle diffèrent.

Le diagnostic sauvegardé montre les pré-tokenizers et les suites d'IDs exactes :
[chat_contract_unicode_diagnostic.json](chat_contract_unicode_diagnostic.json).
Il s'agit d'une limite de cette reconstruction de référence ; cette observation
ne démontre pas une limite linguistique du modèle.

## Périmètre et reproduction

L'adaptateur reste réservé aux messages textuels système/utilisateur/assistant.
Les outils structurés, médias, listes de blocs de contenu et formes de message
non prises en charge sont explicitement refusés. Le support du texte n'implique
aucun support multimodal ni exécution d'outils.

17 tests ciblés du tokenizer passent. Pour reproduire l'audit, employer le
runtime local déjà installé avec `research/real_world/check_chat_contract.py`,
les options `--model`, `--reference-tokenizer` et `--output`. Les chemins exacts,
versions, historiques, rendus, IDs et hashes sont dans le JSON. Le script
désactive les imports Torch/TensorFlow pour cet audit de tokenizer.

Hashes SHA-256 des assets locaux :

- `tokenizer.json` : `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- `tokenizer_config.json` : `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b`
- adaptateur audité : `5004e08f4e157ac088158e2c5c6a10ba56611f6b723a29a5bfe548b94c7cb705`

Source officielle consultée :
[Qwen3.6 tokenizer_config.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/tokenizer_config.json).
L'audit local est lié aux hashes ci-dessus ; il ne prétend pas que les assets
distants de la branche `main` ne puissent jamais changer.
