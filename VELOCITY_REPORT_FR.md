# AirLLM Next — Velocity : génération réelle et limites mesurées

23 octobre 2026, Apple M5, 16 Gio. Recherche issue de la PR #2
(`8775b1e4abb729c02079e91c86e2b9a5cf25aded`). **Le gain exact nouveau confirmé
concerne Granite : environ 8 % au-delà du préchargement déjà optimisé. Une
génération sûre hors mémoire sur le grand modèle n'est pas validée.**

## Résultats de génération native

Les poids officiels `ibm-granite/granite-3.0-1b-a400m-instruct`, révision
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`, occupent 2 669 283 096 octets.
Ils tiennent dans la RAM physique. BF16 source décodé en FP32, même budget total
608 Mio (256 poids, 32 E/S, 64 KV, 256 travail), réserve système 1 536 Mio.
La vérification SHA256 initiale réchauffe le cache OS ; aucun essai n'est vendu
comme débit SSD froid ou test du Fanxiang absent.

Campagne primaire : six prompts, trois répétitions, ordre mélangé, 90 exécutions.
Chaque variante émet 507 tokens réels, dont les sorties arrêtées à EOS. Le temps
inclut préremplissage, calcul, synchronisations et fin des lectures asynchrones.
Construction et encodage des prompts sont rapportés séparément. Tous les tokens
de toutes les variantes correspondent au contrôle apparié ; cela préserve la
sortie gloutonne du modèle, pas une qualité sémantique parfaite de ce petit modèle.

| Variante | Débit génération avec préremplissage | Débit avec construction | Médiane TTFT | Médiane débit décodage |
|---|---:|---:|---:|---:|
| Contrôle, sans préchargement | 4,807 tok/s | 4,801 tok/s | 0,530 s | 5,181 tok/s |
| Préchargement existant | 5,711 | 5,703 | 0,435 s | 6,048 |
| Préchargement + file de 8 experts | 5,771 | 5,763 | 0,379 s | 6,064 |
| Préchargement + import sans copie | **6,166** | **6,157** | **0,371 s** | **6,511** |
| Préchargement + file de 4 + import sans copie | 5,803 | 5,795 | 0,371 s | 6,040 |

Le rapport de débits regroupés sans copie/préchargement est **1,07967×** : 18/18
couples favorables, IC bootstrap apparié exploratoire [1,0650 ; 1,0949]×.
Par rapport au contrôle sans préchargement, le composé atteint 1,28288× ; une
partie du gain vient donc d'une amélioration déjà présente dans #2. Six prompts
sur un seul hôte ne certifient pas un gain universel. Les détails des intervalles,
distributions par prompt et coûts sont dans
[l'analyse indépendante](research/velocity/GENERATION_ANALYSIS.md).

Résultats négatifs : la file de 8 apporte seulement environ 1 % au-delà du
préchargement, avec un intervalle incluant l'absence de gain. Ajouter la file de
4 à l'import sans copie ralentit le débit d'environ 5,9 %. Ces options restent
facultatives, sans modification du routage, de la précision ou des défauts stables.

Confirmation longue distincte : deux répétitions, prompt de 21 tokens produisant
128 tokens et contexte de 402 tokens produisant 123 tokens avant EOS. Les trois
variantes conservent toutes les sorties. Débits regroupés : 5,347 / 5,969 / 6,286
tok/s (contrôle / préchargement / sans copie), soit **+5,31 %** sans copie au-delà
du préchargement. Cette série ne constitue pas un maintien thermique prolongé.

## Coûts, mémoire et E/S

Les pics RSS échantillonnés par le garde externe sont 651 395 072 octets pour
la campagne primaire, 772 603 904 pour la série longue et 812 793 856 avec draft.
Les réserves minimales observées dépassent 4,45 / 7,00 / 6,25 Go respectivement ;
aucune violation du garde dans ces campagnes natives. Le système avait du swap
préexistant ; l'absence de croissance encadrante relevée n'est pas une absence
de swap système.

Sur les 507 tokens primaires de chaque variante, les lectures applicatives sont
411,407 Go pour le contrôle, 415,031 pour le préchargement et 415,738 pour le sans
copie. Leurs compteurs de stockage du processus ne relèvent que 52,7 / 45,2 /
120,3 Mo. **L'import sans copie n'a pas réduit les octets logiques ni démontré
une réduction des lectures physiques.** Il évite une copie d'importation et
réduit un coût de matérialisation dans cet état largement servi par le cache OS.
Ces compteurs ne représentent ni le trafic NAND ni un débit propre au SSD.

Le profil CPU séparé (deux prompts, huit tokens, instrumenté) observe environ
0,647 s de matérialisation avec préchargement contre 0,473 s sans copie en moyenne
par exécution. Les temps de workers peuvent se chevaucher et ne sont pas une
décomposition additive du temps de génération. La conversion BF16 est incluse
dans cette matérialisation ; `dequant_seconds=0` n'en signifie pas l'absence.
L'import DLPack peut vivre hors allocateur MLX : RSS et budgets cache/E/S restent
les contrôles pertinents, indépendamment du pic d'allocateur déclaré.

Temps GPU isolé : indisponible, aucun timestamp de commande Metal instrumenté.
Énergie et température : non mesurées ; `powermetrics` exige ici les privilèges
administrateur. `pmset` ne rapporte pas d'avertissement enregistré, ce qui n'est
pas une mesure de température. Aucun état privilégié du système n'a été changé.

## Draft incrémental : implémenté, résultat global négatif

`IncrementalModelDraft` conserve ses KV, réconcilie le préfixe accepté et
tronque/rejoue après rejet. `IncrementalReencodedModelDraft` ajoute les contrôles
de raccord entre tokenizers. Les mappings Granite et SmolLM2 diffèrent : ils ne
sont jamais assimilés. Chaque proposition reste vérifiée par la cible.

La série Metal compare trois prompts, deux répétitions et un plafond de 32
tokens, à somme de budgets égale. Chaque variante émet 146 tokens, tous identiques
au contrôle. Résultats regroupés : contrôle 5,197 ; préchargement 5,788 ; ancien
draft 4,312 ; draft incrémental 4,496 tok/s. Le nouveau draft gagne 4,26 % sur
l'ancien, mais reste **22,32 % sous le préchargement sans draft**. Il ne devient
pas le chemin par défaut. Sa résidence de poids reste limitée à 64 Mio dans ce
contrôle ; d'autres couples de modèles demandent leur propre campagne.

## Quantification : Q8 plus prudent sur les mêmes labels

Une nouvelle conversion des 1 536 matrices experts en Q8, groupes de 64,
produit 1 358 954 496 octets de payload. Les autres poids restent originaux.
Les anciennes variantes Q4/mixte et leurs résultats ne sont pas écrasés.

Le contrôle qualité recalcule exactement la référence précédente sur les mêmes
512 labels WikiText-2 immuables. Perplexité de référence : 13,38435 ; experts Q8 :
13,37478 (rapport 0,999285). Accord top-1 : **508/512**, soit 99,21875 %. L'ancien
mixte Q4/Q8 avait un rapport de perplexité 1,052505 sur ces mêmes labels.
Le Q8 réduit donc fortement la déviation observée ici, mais **ne préserve pas
exactement les logits ni tous les argmax**. Une variation de -0,071 % sur seulement
512 labels ne prouve pas une amélioration générale de qualité. Aucun gain de
génération de cette nouvelle variante Q8 n'est revendiqué : cette série mesure
la qualité par positions connues, séparément du débit.

## Modèle déjà installé : test effectué, limite sûre explicite

Après l'indication de l'utilisateur, l'inventaire a retrouvé le modèle Ollama
`qwen3.6:latest`. Son GGUF Q4_K_M occupe exactement **23 938 321 664 octets**,
donc plus que les 17 179 869 184 octets de RAM physique. Le total de tag de
23 938 333 577 octets inclut aussi licence et paramètres. SHA256 intégral vérifié :
`f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`.
Licence Apache-2.0, famille GGUF `qwen35moe`, 40 couches, 256 experts dont huit
activés, modèle officiel Qwen3.6 35B-A3B. Le nombre de paramètres GGUF inclut
aussi des composantes vision/MTP ; il ne remplace pas le compte actif textuel.

Le premier essai Ollama CPU émet « Le » après 40,24 s : chargement 24,99 s,
préremplissage 15,20 s. **Ce n'est pas une validation sûre** : le swap passe de
2 071 396 352 à 20 334 706 688 octets. La surveillance détecte un dépassement,
mais son ancien filtre attendait un processus `runner` alors que cette version
lance `llama-server` ; elle n'a donc pas arrêté assez tôt ce premier chargement.
Le modèle est déchargé à la fin de la requête. Ce défaut a été corrigé et n'est
pas masqué par le token émis. Les journaux révèlent un tampon `CPU_REPACK` de
21 086,60 Mio avec le chargement Ollama `load_mode=none`.

Les contrôles suivants possèdent directement leur sous-processus et le
terminent sur seuil mémoire/swap, sans toucher au service Ollama ni aux fichiers
personnels. Les essais `mmap`, sans repack ni warmup, sont arrêtés avant génération
avec des hausses de swap observées de 175 et 300 Mo. Le prototype local qui
supprime uniquement les conseils WILLNEED passe son petit test d'API et son
injection sur une copie privée du binaire ; le vrai démarrage est toutefois
arrêté après une hausse de 280 Mo, **avant tout appel WILLNEED intercepté**.
L'hypothèse n'est donc pas établie comme cause du blocage observé et aucun gain
n'en est annoncé. Tous les échecs sont conservés. Les binaires installés restent
inchangés et aucun worker de ces essais n'est laissé actif.

En contrôle utile, le modèle déjà installé `qwen3:8b` génère 32 tokens via Ollama :
26,27 tok/s selon le compteur de décodage serveur, 8,77 tok/s chargement compris,
TTFT 2,44 s, pic RSS échantillonné 5 334 073 344 octets, garde non déclenché.
Ce sont des résultats **Ollama**, distincts du moteur natif AirLLM.

Le backend natif n'a pas de lecteur GGUF Q4_K/Q6_K validé pour le modèle hybride
installé. Une conversion BF16 complète dépasserait l'espace disque disponible.
Voir [la faisabilité détaillée](research/velocity/GGUF_FEASIBILITY.md). Un autre
vrai MoE officiel MLX 6 bits de 23 853 636 272 octets a été sélectionné avec garde
disque et adaptateur borné ; son téléchargement lent a été arrêté après la
découverte du modèle installé. Les fragments restent conservés, et **ne sont
pas un checkpoint complet ni une validation**. Son adaptateur ne certifie pas
la compatibilité Qwen3.6 : ce sont des architectures différentes.

## Validation, livraison et reproduction

- Suite finale : **994 tests réussis, un ignoré, 11 sous-tests réussis**. Contrôles
  Metal ciblés : 36 réussites sur le candidat initial ; tests Qwen finaux :
  16 réussites CPU, dont quantification 6 bits/groupes 128/échelles BF16 et
  comparaison aux équations Transformers. Les fixtures minuscules restent
  identifiées comme telles.
- 132 exécutions natives réparties entre les campagnes de génération principale,
  draft, longue et profil CPU, avec contrôles de tokens et garde externe. Ce
  n'est pas une génération unique de 4 673 tokens ni un test thermique prolongé.
- Les sources primaires sont hashées dans chaque rapport. Les 38 empreintes de
  la première campagne correspondent au commit local `574feec`, arbre publié
  `1cf33e82c57cf09b66c0235295a9e6a5e1a83c68`, commit GitHub `d66ce9f`.
  La révision suivante durcit le nouveau backend Qwen sans modifier les calculs
  Granite exécutés. Les empreintes, et non le seul HEAD relevé au démarrage,
  identifient le code effectivement mesuré.
- [Protocole](research/velocity/PROTOCOL_FR.md), [audit des coûts](research/velocity/LOAD_AUDIT.md),
  [sources primaires et limites des publications](research/velocity/PRIMARY_RESEARCH.md).
- `scripts/validate_velocity.sh` reproduit les tests sans téléchargement.
  `benchmarks/benchmark_velocity.py` reproduit les ablations avec un chemin
  explicite vers le checkpoint ; `--variants reference prefetch zero-copy`,
  `--cases 0 1 2 3 4 5 --repeats 3 --tokens 32`. Les commandes et budgets exacts
  sont dans les JSON. L'import sans copie est aussi exposé par
  `airllm-next run … --zero-copy-weights` et reste facultatif.
- Les JSON publics sont compressés sans perte et accompagnés d'un manifeste
  SHA256 dans `evidence-velocity/`. Restaurer par
  `python benchmarks/archive_velocity_evidence.py --restore`. Les poids, fichiers
  texte du corpus et binaires compilés ne sont pas inclus.

Les niveaux A–F hors mémoire restent **non atteints dans cette validation**.
Les résultats ne démontrent ni modèle 300–700B, ni SSD externe, ni 10–15 tok/s
hors mémoire. La livraison est une PR de recherche dépendante de #2, sans fusion
dans `main`, avec les gains limités confirmés et les blocages conservés.
