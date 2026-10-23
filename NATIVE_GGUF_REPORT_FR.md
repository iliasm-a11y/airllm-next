# AirLLM Next — inférence native GGUF sur M5 / 16 Gio

**23 octobre 2026. Le GGUF Qwen3.6 installé de 23,94 Go fonctionne désormais
dans le moteur natif AirLLM Next, jusqu'à 128 nouveaux tokens, avec un pic RSS
échantillonné de 637 Mo sur cet essai et aucune croissance de swap observée.**
La campagne primaire atteint 0,586 token/s en comptant le processus supervisé,
ses imports, son initialisation et sa génération, à entrées déjà tokenisées.
Le seuil de 1 token/s de bout en bout n'est pas confirmé.

Le produit reste AirLLM Next. Il s'agit d'une branche de recherche, pas d'une
version stable 8.0. Elle part de la tête réelle de la PR #3,
`5992ec7778364162dff416a27715f966bbcb3b31`, sans fusion dans `main`.
Le [commit moteur publié](https://github.com/iliasm-a11y/airllm-next/commit/81b6a91ddf5760576226e4cb91e22a593ad27d60)
possède exactement l'arbre du candidat local. Les SHA256 des sources de chacun
des 24 essais correspondent à ce commit, vérifiés dans
`evidence-native-gguf/source-validation.json`.

## Checkpoint réel et composants intégrés

Le fichier installé contient **23 938 321 664 octets**, contre
17 179 869 184 octets de mémoire physique. Son SHA256 intégral a été revérifié
avec un tampon de 4 Mio et un processus supervisé :
`f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`.
Aucun checkpoint n'a été téléchargé, dupliqué ou converti intégralement.
Seuls les petits fichiers du tokenizer officiel ont été obtenus pour comparaison.

Le lecteur trouve 1 194 tenseurs : 349 Q4_K, 94 Q6_K, 544 F32 et 207 F16.
L'architecture `qwen35moe` comporte 40 couches, 256 experts dont huit sélectionnés
par token, un expert partagé et une alternance de trois blocs DeltaNet pour
un bloc d'attention complète. Le format Q4_K_M mélange plusieurs types de
tenseurs ; ce n'est pas une quantification affine MLX uniforme.

Les nouveaux composants sont intégrés au paquet et à sa commande existante :

- `gguf.py` : métadonnées et index sans mmap, contrôles des offsets/tailles,
  erreurs et fichiers tronqués, lectures positionnelles de lignes et d'experts,
  codecs Q4_K/Q6_K/F16/F32, réservations concurrentes avant allocation.
- `gguf_qwen.py` : adaptation des conventions GGUF, normes, A et ordre des
  têtes V ; réutilisation des équations, sessions et limites Qwen existantes.
  Seuls les experts effectivement routés et les tuiles courantes sont chargés.
- `gguf_metal.py` : produits directement sur blocs Q4_K/Q6_K compactés,
  accumulation FP32, sans matrice complète déquantifiée.
- `gguf_tokenizer.py` : vocabulaire/merges embarqués, normalisation NFC,
  tokens ajoutés et template texte comparés au tokenizer officiel.
- `experiment_guard.py` et `gguf_cli.py` : admission préalable, limites de RSS,
  réserve système, pression macOS, croissance du swap et arrêt du groupe
  d'expérience ; protection des inputs contre les chemins de sortie aliasés.

Le cache adaptatif conserve son budget. La taille des tuiles et la demi-vie de
ses observations sont des options explicites. Les chemins existants restent
disponibles. Les binaires Ollama installés n'ont pas été modifiés ni exécutés
pour ces mesures natives.

## Mesures de génération réelle

La campagne primaire comprend **trois prompts, deux répétitions et quatre
variantes**, en ordre mélangé : 24 exécutions et 36 nouveaux tokens par variante.
L'arrêt EOS raccourcit le cas arithmétique. Chaque variante conserve les mêmes
tokens. Tous les budgets sont identiques : cache 256 Mio, E/S 64 Mio,
KV/états 192 Mio, travail 128 Mio et réserve système 3 Gio.

| Variante | Génération avec préremplissage | Décodage | Processus supervisé complet | TTFT médian après initialisation |
|---|---:|---:|---:|---:|
| Référence GGUF déquantifiée, tuiles 2048 | 0,157 tok/s | 0,198 tok/s | 0,154 tok/s | 8,722 s |
| Metal compacté, tuiles 512 | 0,333 | 0,470 | 0,324 | 4,446 s |
| **Metal compacté, tuiles 2048** | **0,621** | **0,942** | **0,586** | **3,119 s** |
| Metal 2048, cache à décroissance plus lente | 0,634 | 0,947 | 0,598 | 2,278 s |

Les débits sont des rapports de sommes. La génération compte préremplissage,
lectures, préparation, calculs, synchronisations et suivi interne. La colonne
processus ajoute imports, initialisation et arrêt supervisé. Les entrées de cette
campagne sont déjà tokenisées : le coût de construction du tokenizer est séparé.
L'essai CLI texte `cli-chat4.json` inclut ce coût et produit « Le ciel apparaît bleu ».

Le chemin Metal 2048 gagne **3,97× en génération** et **4,77× en décodage** face
au nouveau chemin GGUF de référence, sur ces six paires. Le passage 512→2048
gagne respectivement **1,86×** et **2,00×**. Le premier gain regroupe kernel,
transport et représentation du cache : il ne mesure pas un kernel GPU isolé.
Ce ne sont ni des gains mesurés sur Granite, ni une comparaison de débit avec
l'ancien essai Ollama à swap excessif.

Le pic RSS de la variante Metal 2048 dans la campagne primaire est 711 Mo ;
celui de la variante cache expérimental est 748 Mo. Aucun garde n'est déclenché,
aucune croissance de swap n'est observée dans les 24 essais. Le système utilise
déjà environ 3,1 Go de swap au début : **croissance nulle ne signifie pas swap nul**.

## Générations prolongées et résultats négatifs

Ces essais distincts ne sont pas ajoutés opportunément aux 24 paires.

| Essai | Tokens effectivement produits | Génération avec premier calcul | Décodage | Pic RSS échantillonné | Croissance swap |
|---|---:|---:|---:|---:|---:|
| Metal 2048, 64 tokens | 64 | 0,902 tok/s | 0,913 tok/s | 646 Mo | 0 |
| Metal 2048, 128 tokens | 128 | 0,988 | 1,003 | 637 Mo | 0 |
| Même chemin, cache lent, 128 tokens | 128 | 0,955 | 0,969 | 659 Mo | 0 |

Les 128 tokens des deux variantes sont identiques ; les 64 premiers correspondent
aussi à l'essai de 64 tokens. La continuation brute « Bonjour » devient du code
VBA très répétitif : elle démontre la stabilité du calcul et des états, pas une
qualité conversationnelle générale ou une charge représentative de tous les usages.
Les états/KV atteignent environ 146 Mo de payload comptabilisé, sous leur limite.

La demi-vie de cache de 32768 requêtes, au lieu de 1024, réduit les lectures
logiques de **12,66 %** sur 128 tokens : 219,48 contre 251,29 Go. Pourtant le
débit baisse de **3,36 %**. Dans la campagne courte, son gain groupé n'est que
2,1 % en génération et 0,5 % en décodage. **Cette option reste facultative** :
réduire les octets ne suffit pas à accélérer le moteur.

Le contrôle séparé `pread_nocache`, huit tokens, conserve les sorties et mesure
0,544 tok/s avec premier calcul, 0,547 en décodage, 612 Mo de pic RSS et aucune
croissance du swap. Il lit 15,53 Go logiques et relève 5,20 Go de lectures de
stockage du processus. `F_NOCACHE` n'est pas une purge du cache OS ; cet essai
ne certifie pas un SSD froid ni un débit NAND.

## Exactitude et validation

**Suite complète finale : 1 087 tests réussis, un ignoré et 11 sous-tests réussis.**
La suite de départ de la PR #3 a d'abord été reproduite à 994 réussites.
Les commandes, journaux et résultats XML locaux sont conservés. Aucun résultat
de CI distante n'est substitué à ces tests locaux.

Les contrôles numériques distinguent quatre niveaux :

1. **Codecs indépendants** : 28 échantillons du vrai fichier, égalité FP32
   bit à bit avec le Python officiel `gguf` et les décodeurs C GGML indépendants.
2. **Kernels** : petits blocs et vraies matrices contre déquantification
   indépendante ; contrôles de budgets avant import et traitement des experts.
3. **Architecture miniature** : logits, plusieurs tokens, CPU/GPU, permutation
   V non involutive, interruptions et restauration des états contre Transformers.
4. **Checkpoint réel** : les équations officielles Transformers sont exécutées
   dans Torch avec un modèle créé sur `meta` et des transports de poids bornés.
   Le routeur et le calcul des experts restent les classes officielles ; ni le
   calcul MLX ni ses adaptateurs de noms/configuration ne sont réutilisés.
   Les codecs seuls sont partagés et ont leurs contrôles indépendants précédents.

Les préfixes des 24 essais correspondent à Torch jusqu'à quatre positions,
ou deux pour le cas arrêté à EOS. Trois vecteurs complets de 248 320 logits,
aux dernières positions ainsi couvertes, ont des erreurs maximales de
3,81×10⁻⁶ / 2,10×10⁻⁵ / 1,07×10⁻⁵, sous les tolérances fixées
`atol=2e-4, rtol=5e-5`. Le premier token exploratoire est également vérifié
indépendamment. **Les 128 positions ne sont pas toutes recalculées dans Torch** ;
leur contrôle est une identité entre variantes natives.

L'exactitude d'implémentation ne rend pas une réponse sémantiquement vraie :
le prompt brut `2+2=` produit ici `5` puis EOS dans les deux moteurs.
Aucun score de qualité du checkpoint BF16 original n'est revendiqué.

## E/S, limites et objectifs

Dans la campagne primaire Metal 2048, les lectures logiques atteignent 86,71 Go,
contre 26,16 Go relevés par le compteur de stockage du processus. Le temps CPU
est environ 0,721 s par token émis, coût du prompt inclus. Les temps matmul
instrumentés incluent imports éventuels, lancement et synchronisation ; les
timestamps GPU isolés, l'énergie et la température ne sont pas mesurés.

Le cache OS n'est pas contrôlé. Le contrôle d'intégrité intégral et ses compteurs
sont une preuve sur les octets du fichier, pas une mesure de bande passante NAND.
Il a été exécuté pendant la validation finale des tests, hors campagne d'inférence.
Les pics RSS sont échantillonnés toutes les 50 ms ; les réservations du lecteur,
du contexte et de MLX agissent en complément avant les allocations prévues.

Le parcours froid textuel demande environ **1,965 Go de poids par position**,
dont 0,653 Go pour les experts actifs. Sans résidence ni amortissement, 2 tok/s
nécessiteraient déjà 3,93 Go/s de délivrance des poids et 8 tok/s, 15,72 Go/s,
avant les coûts de calcul. Ce calcul conditionnel n'est pas une mesure du SSD.

| Objectif | État de cette campagne |
|---|---|
| Initialisation sous 16 Gio | Vérifiée avec limites explicites |
| Premier token natif correct | Vérifié contre calcul indépendant |
| 32 tokens sans saturation | Atteint ; essais poussés à 128 |
| 0,5 tok/s soutenu | Observé, dont 0,586 tok/s processus complet dans la campagne à IDs fournis |
| 1 tok/s de bout en bout | Non confirmé ; 1,003 en décodage sur un seul essai long |
| 2, 4 ou 8 tok/s | Non atteints |
| 300–700B | Non validé et sans extrapolation de cette mesure |

Le chemin reste **texte uniquement** : vision, MTP et spéculation sur ce backend
ne sont pas exécutés. La vérification multi-tokens et les drafts des travaux
antérieurs ne sont pas présentés comme des accélérations du nouveau GGUF.
Le SSD externe, les longs contextes, une campagne thermique prolongée et une
évaluation générale de qualité restent à tester. Aucun mécanisme n'est présenté
comme une invention originale sans antériorités.

## Reproduction et preuves

- [Protocole complet](research/native_gguf/PROTOCOL_FR.md),
  [analyse calculée](research/native_gguf/ANALYSIS_FR.md),
  [sources primaires et antériorités](research/native_gguf/PRIMARY_SOURCES.md),
  [analyse des octets et du cache](research/native_gguf/ROOFLINE.md).
- `scripts/validate_native_gguf.sh` : tests sans téléchargement de poids.
- `benchmarks/campaign_native_gguf.py` : ablations mélangées à budgets égaux.
- `benchmarks/reference_gguf_torch.py` : calcul indépendant borné, à superviser.
- `benchmarks/analyze_native_gguf.py` : reconstruction des tableaux sans modèle.
- `evidence-native-gguf/` : JSON, logs, gardes, empreintes et résultats négatifs.
  Les vecteurs binaires de logits et XML restent locaux, hashés dans le manifeste.
  Aucun poids, secret, corpus personnel ou binaire Ollama n'est publié.

Exemple explicite du chemin rapide mesuré :

```sh
export PYTHONPATH="$PWD/air_llm"
python -m airllm.nextgen.cli gguf-run /chemin/du/modele.gguf \
  --prompt 'Explique en deux phrases pourquoi le ciel est bleu.' \
  --kernel metal --row-chunk 2048 --max-new-tokens 32 --report run.json
```

La commande lance son processus supervisé par défaut. Les défauts génériques de
précision, de cache et de backend des autres architectures ne sont pas remplacés.
