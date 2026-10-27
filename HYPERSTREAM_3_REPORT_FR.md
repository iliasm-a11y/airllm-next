# HyperStream 3 M5 — implémentation expérimentale, campagne réelle différée

27 octobre 2026. **Aucune accélération H3 n'est démontrée. Les objectifs de 3, 5 ou 10 tokens/s ne sont pas atteints par une mesure de cette livraison.** Le code TensorOps est écrit mais n'a pas été compilé/exécuté sur GPU. La fidélité numérique du checkpoint réel reste à établir.

Branche : `research/hyperstream-3-m5-native`, issue exactement de la PR #11 publiée au commit `5e8234181e736c0319a64df25e6813e92dd3e6da`. Les PR #10/#11 et `main` ne sont ni fusionnés ni modifiés.

## Changements du moteur

### Produits matriciels groupés

`gguf_m5.py` ajoute deux chemins facultatifs aux produits ordinaires et aux experts déjà sélectionnés :

- `tensorops` : tuiles **16×16×32**, décodage des blocs GGUF Q4_K/Q6_K d'origine dans 4 Kio de mémoire threadgroup, opérandes/accumulateurs FP32, `matmul2d_descriptor(..., relaxed_precision=false)` et compilation MLX en mode `safe`. Aucun nouveau checkpoint, passage FP16, réencodage ou quantification d'activation n'est ajouté.
- `reuse` : une déquantification sert à 2, 4, 8 ou 16 tokens dans les registres, avec l'ordre des additions par lane et la réduction SIMD du chemin existant. Ce n'est pas TensorOps.

Le chemin unitaire de décodage conserve le kernel précédent. Sur chaque nouvelle forme matricielle, le dispatcher compare un échantillon réel aux résultats du kernel précédent (`atol=rtol=1e-4`), puis mesure quatre paires de temps synchronisés AB/BA après compilation/échauffement. Il ne conserve le candidat que si la médiane observée gagne plus de 3 %. Compilation, comparaison, calibrations, synchronisations et replis restent dans la latence totale de la requête. Cette sélection locale n'est **pas** une validation de tous les futurs logits ou de la séquence complète.

Une absence d'API, une erreur de compilation/exécution différée, un échec numérique, un gain insuffisant ou une réserve insuffisante conserve le kernel précédent. Les buffers de calibration sont chargés conservativement à hauteur de six sorties supplémentaires et d'une tuile threadgroup ; le budget commun des pipelines est transmis au dispatcher. Les erreurs du kernel de référence ne sont pas masquées.

La synchronisation imposée pour détecter les erreurs MLX différées peut réduire le chevauchement du pipeline H1. Elle doit être mesurée. Aucun backend n'est activé par défaut et aucune sélection issue d'un faux MLX de test n'est utilisée en production.

L'audit officiel de MLX **0.32.4** confirme la compilation Metal à l'exécution : l'absence de `xcrun metal` n'est donc pas, à elle seule, un blocage de TensorOps. Les références Apple, MLX et llama.cpp, leurs différences et les licences sont dans [SOURCES.md](research/hyperstream3/SOURCES.md). La PR llama.cpp #29869 concerne notamment un repli SIMD à quelques lignes ; ses gains sur d'autres machines ne sont pas attribués à AirLLM/M5. Le nouveau code reprend les décodeurs GGML déjà présents, pas une réécriture intégrale des kernels externes.

### Vérification spéculative et modèles hybrides

`NativeGGUFQwenModel.generate_speculative` intègre le vérificateur glouton existant au modèle GGUF. Le générateur facultatif par défaut est le prompt lookup local ; un petit modèle déjà chargé ou un callable peut aussi proposer des IDs compatibles. Aucun modèle n'est téléchargé. Les profondeurs proposées sont 2, 4, 8 et 16 ; une vérification traite le token en attente et jusqu'à cette profondeur de propositions.

Chaque token retenu est l'argmax du grand modèle sur le préfixe vérifié. Le cache Qwen hybride ne peut pas être tronqué comme un simple KV append-only : le nouvel adaptateur restaure les banques KV, convolution et DeltaNet depuis un snapshot, puis rejoue uniquement les entrées acceptées. Ce rejeu est une vraie exécution du grand modèle, avec temps, experts et lectures comptés. L'acceptation intégrale ne demande pas ce rejeu. Le contrôleur adaptatif existant compare le coût total observé et revient au décodage ordinaire si la spéculation ralentit.

Les snapshots restent dans `kv_bytes`. Un refus des deux budgets de snapshots explicitement reconnus provoque un repli depuis le prompt initial : anciens états et références de traceback sont libérés avant le nouveau cache. Travail abandonné et nouveau préfill restent comptés. Une autre erreur mémoire n'est pas transformée en repli silencieux. NaN/Inf sont rejetés, y compris dans le repli.

**L'équivalence gloutonne à l'exécution sérielle réelle n'est pas démontrée.** Le regroupement change l'association des calculs flottants et peut changer un argmax proche. Les contrôles de flux sur cible récurrente miniature et le principe de vérification ne remplacent pas le test du vrai checkpoint.

### Streaming exact des experts

`read_packed_experts_into` lit directement les experts demandés dans une seule allocation finale `[experts, lignes, octets]`, au lieu d'allouer chaque expert puis le recopier dans ce groupe. Il ne fusionne que les sélections contiguës à la fois dans le fichier et dans la destination, sans lire les trous ni des experts non demandés. Les limites de lecture, les allocations propriétaires, les vues, les lectures partielles et les erreurs restent contrôlées.

Le transport groupé **contourne explicitement le cache individuel des tuiles** : il peut donc augmenter les lectures ultérieures malgré les copies économisées. Il reste facultatif et son gain doit être mesuré. Le double buffering et le pipeline gate/up déjà présents sont conservés ; aucun nouveau chevauchement SSD/GPU n'est déclaré prouvé. Il n'y a ni nouvelle disposition sur disque, ni suppression d'expert, ni attention approchée. L'analyse détaillée est dans [IO_ANALYSIS_FR.md](research/hyperstream3/IO_ANALYSIS_FR.md).

## Les 178 192 chargements historiques

Les compteurs relus dans la preuve H2 se décomposent exactement ainsi :

`178 192 = 3 × 56 783 utilisations d'experts + 7 843 projections ordinaires`.

Environ **95,60 %** concernent donc gate/up/down des experts. Le payload direct moyen est d'environ **752 Kio** : il ne s'agit pas de 178 000 minuscules lectures scalaires. Ces compteurs couvrent **toute la génération**, et non le seul préfill. Ils ne permettent pas de reconstruire combien d'experts étaient adjacents ni la part d'attente SSD pure.

Le cas historique de 364 tokens utilisait déjà des blocs de 16 : 23 passages complets, TTFT 110,699902 s et 135 518 514 304 octets logiques au premier forward. Les temps historiques inclusifs se recouvrent ; leur somme n'est pas une décomposition CPU/GPU/SSD. Les blocs 32/64/128 et la réutilisation par vérification peuvent réduire les répétitions, mais les unions d'experts et le rejeu doivent être comptés.

## Conditions matérielles et refus

MacBook Air M5, 16 Gio, macOS 27.2 ; MLX 0.32.4, NumPy 2.5.3. Le checkpoint historique présent fait **23 938 321 664 octets**, donc dépasse la RAM physique. La lecture de son contenu complet n'a pas été autorisée par la garde actuelle.

Pendant **601,12 secondes**, les **121 relevés** indiquent tous une pression macOS de niveau **2**, jamais le niveau normal 1. Mémoire disponible : **2 467 069 952 à 3 767 173 120 octets**. Swap initial/final : **2 953 969 664 / 2 920 415 232 octets**. Cette variation est système et n'est pas attribuée à un moteur d'inférence. Les processus node/npm et Codex/ChatGPT contribuent aux RSS observées ; leur somme inclut des pages partagées et ne mesure pas la propriété des pages compressées ni une causalité complète.

Les admissions H1, H2, TensorOps, réutilisation, transport groupé, spéculation et combinaison ont toutes été refusées **avant le lancement d'un worker**. La combinaison n'a subi qu'un contrôle d'admission, pas un essai de composants non validés. Le calcul SHA-256 intégral préalable a lui aussi été refusé. Il n'existe donc **aucun nouveau hash intégral avant/après, aucune sortie réelle H3 ni aucun benchmark complet H3**.

Réserve 3 Gio, RSS 2 Gio, croissance swap 128 Mio, payload d'admission 768 Mio, sondage 50 ms et plafond de 32 processus possédés restent inchangés. Aucune application n'a été fermée et aucun seuil n'a été abaissé. Les journaux et refus sont conservés dans `research/hyperstream3/evidence/`.

**Incident de sélection des tests, déclaré et corrigé :** un fichier hérité inclus dans la première sélection dite portable a exécuté 16 tests GPU de l'ancien kernel sur des matrices synthétiques minuscules (4 experts, 13 lignes, largeur 512), sans garde d'admission. Aucun checkpoint réel ni nouveau kernel H3 n'a été exécuté par ces tests. Leur journal est conservé séparément et exclu des preuves de campagne M5. Ces tests exigent désormais un opt-in explicite et une admission normale ; la sélection finale les a différés. L'absence de mesure continue pendant cet incident interdit toute conclusion de stabilité mémoire à son sujet.

## Limite physique : ce que les chiffres permettent de dire

La référence historique H1 reste **1,584 token/s**, pas une nouvelle mesure. Elle demandait environ **1,964699 Go logiques par token décodé**. À trafic inchangé, 3, 5 et 10 tokens/s demanderaient respectivement **5,894 / 9,823 / 19,647 Go/s logiques**. Ce calcul concerne un domaine logique ; il ne mesure pas le SSD.

La lecture intégrale historique distincte indiquait environ 4,39 Go/s logiques et un volume substantiel imputé par macOS au stockage. D'autres essais avaient surtout lu le cache de fichiers. Employer 4,39 Go/s avec le trafic historique donne un **scénario conditionnel** d'environ 2,23 tokens/s, pas un plafond physique démontré : les régimes de cache et les motifs d'accès diffèrent. Il serait incorrect d'en conclure que 3 tokens/s est impossible, ou que 10 tokens/s est accessible.

`estimate_hyperstream3_limit.py` exige le débit fourni par l'opérateur et distingue trafic logique, cache supplémentaire, émissions vérifiées par bloc, coût des unions/rejeux, draft et calcul. Aucun débit SSD par défaut n'est inventé. Un disque externe physique n'est pas énuméré ; `/Volumes/SSD` est un lien vers le volume interne. Les débits actuels interne/externe, temps GPU purs, plafond physique et endurance thermique restent non mesurés. Le harnais ajoute le temps CPU utilisateur/système et conserve les attentes MLX côté hôte ; il ne les rebaptise pas temps GPU.

## Vérification et reproduction

Résultats finaux dans les journaux XML/stdout :

| Ensemble | Réussis | Sous-tests réussis | GPU différés |
|---|---:|---:|---:|
| H3 : kernels/replis, allocations, planificateur, CLI, cibles récurrentes et falsification des preuves | 198 | 49 | 64 |
| Contrôles hérités sélectionnés, après correction | 217 | 60 | 16 |
| Total des deux ensembles distincts | **415** | **109** | **80** |

Ces contrôles valident des invariants locaux et des protocoles ; ils ne prouvent pas la fidélité ou la vitesse du checkpoint complet. La suite générale Torch/MLX et un essai de stabilité prolongé n'ont pas été exécutés. La CI Ubuntu ajoutée vérifie les contrôles H3 sans MLX/Torch ; son état publié doit être lu séparément.

Le nouveau vérificateur indépendant `verify_gguf_speculation.py` n'importe ni moteur, ni NumPy, ni analyseur de performances. Il vérifie les fichiers FP32 bruts, leurs hashes, formes et finitude, reconstruit les préfixes consommés y compris rejets/rejeux, compare tous les logits des positions émises, argmax, EOS et routes acceptées de chaque couche. Les branches rejetées ne sont pas confondues avec le préfixe final. Il ne certifie pas les gardes, l'intégrité complète des poids, les états récurrents cachés ou les gains de temps : ces preuves restent séparées.

[Le protocole de reprise](research/hyperstream3/README.md) décrit les 16 expériences séparées, les paires AB/BA, les hashes complets, les conditions d'admission et les limites de validation. Les combinaisons attendent la validation de leurs composants. **La mission d'accélération réelle reste inachevée tant qu'une fenêtre mémoire normale ne permet pas compilation, fidélité réelle et mesures appariées.**
