# AirLLM Next — validation matérielle M5

27 octobre 2026. Branche `research/hyperstream-m5-validation`, issue de la PR #12,
commit `3b137621d69f3549a85b7d0cc20460bd07b11fb9`. Aucune quatrième génération
d'optimisations n'est ajoutée. Les chemins moteur, poids, précision et gardes
hérités restent inchangés.

## Résultat réellement obtenu

Le diagnostic mémoire et le banc autonome sont livrés et testés sur leurs
contrôles portables. L'admission matérielle reste refusée : **aucun nouveau
kernel n'a été compilé ou exécuté sur le GPU, aucun token du checkpoint réel
n'a été généré et aucun gain H2/H3 n'est démontré dans cette session**.
Les objectifs de 3, 5 et 10 tokens/s restent exploratoires.

La mission ne s'est pas limitée à une nouvelle attente de dix minutes : le
banc refuse immédiatement par défaut, enregistre le refus, et peut être lancé
depuis un Terminal indépendant de Codex après libération volontaire des ressources.
Une fenêtre d'observation courte et explicitement bornée reste facultative.

## Blocage mémoire établi

Le relevé principal `research/m5_validation/evidence/memory-verified.json`, pris
à **13:15:45 heure de Paris**, rapporte :

| Observation | Valeur |
|---|---:|
| Mémoire physique | 16 Gio |
| Pression macOS | 2, avertissement |
| Mémoire disponible | 3,084 Gio |
| Swap utilisé | 3,142 Gio |
| Compresseur : occupation physique | 5,919 Gio |
| Même contenu avant compression | 10,077 Gio |
| Mémoire câblée | 3,452 Gio |
| Admission du modèle : réserve + payload | 3,75 Gio |
| Déficit à cette admission | environ 682,4 Mio |
| Marge supplémentaire de la fenêtre de reprise | 256 Mio |

Les catégories se recouvrent et ne s'additionnent pas. Les pages adossées à des
fichiers ne sont pas toutes un cache immédiatement récupérable. Le pourcentage
de `memory_pressure -Q` ne remplace ni la mémoire disponible ni le niveau de
pression retourné par macOS.

L'outil macOS `footprint` confirme des empreintes notables de Codex/ChatGPT,
dont environ 1,375 Go pour le renderer et 1,239 Go pour le serveur principal.
Ces comptes ne prédisent pas exactement la mémoire libérée par leur fermeture.
Les sommes RSS de Node/npm et Codex sont conservées comme indicateurs non
exclusifs ; elles ne constituent pas une attribution de la mémoire physique.
Aucun worker d'inférence identifiable n'a été trouvé. Aucun arrêt de processus,
fermeture d'application, redémarrage ou purge de cache système n'a été effectué.

L'inventaire n'importe pas MLX et n'ouvre pas de contexte GPU. Les compteurs MLX
étant locaux au processus, une consommation GPU globale et un cache moteur
résiduel ne sont pas inventés : ils restent `null` en l'absence de worker possédé.
Les arguments des autres processus sont retirés des preuves destinées à GitHub.
Les originaux privés restent conservés localement, avec leur empreinte SHA-256.

## Kernels : API contrôlée, exécution encore différée

MLX **0.32.4** est présent dans l'environnement local. Les signatures installées,
le code officiel du tag et les en-têtes du SDK ont été inspectés. Les paramètres
`math_mode=safe`, les types FP32 et les tenseurs inline sont présents. Ces constats
ne prouvent pas qu'une spécialisation TensorOps compile sur le M5.

Le compilateur Metal et `xctrace` hors ligne ne sont pas disponibles via `xcrun`.
La compilation JIT de MLX doit donc être testée par le worker GPU lorsque
l'admission sera possible. Aucun correctif spéculatif n'est appliqué à
`gguf_m5.py` sans erreur effectivement reproduite.

`benchmarks/validate_m5_kernels.py` fournit ce test : Q4_K/Q6_K, formes groupées
et ordinaires, blocs de tokens 16/32/64/128, largeurs 256/512/1024 et bords
partiels. TensorOps et les tuiles de réutilisation 2/4/8/16 sont comparés au
décodeur scalaire et à l'ancien kernel. La tolérance est fixée avant mesure à
`atol=1e-4, rtol=5e-5`. Les candidats indisponibles, inexacts ou plus lents sont
conservés comme résultats négatifs. Le chemin historique reste disponible.

Le payload léger est adapté à son empreinte : 256 Mio, plafond RSS 512 Mio.
La réserve de 3 Gio, la pression normale et la croissance maximale du swap de
128 Mio restent obligatoires. Le point d'entrée privé vérifie également
l'admission avant l'import MLX. Le premier essai réel a été refusé à pression 2,
avant tout worker : voir `kernel-admission.json` et sa garde.

Les mesures futures distinguent compilation/première exécution, temps hôte
synchronisé, temps CPU et allocations MLX. La durée GPU pure reste `null` :
l'API Python inspectée n'expose pas les timestamps nécessaires. Une capture
Metal facultative ne devient pas automatiquement une mesure de temps GPU.
Voir `research/m5_validation/KERNEL_AUDIT_FR.md` pour les sources et limites.

## Banc autonome, preuve et reprise

Les commandes `prepare`, `run` et `verify` de `benchmarks/m5_validation.py`
figent le checkpoint attendu, les commits, les sources, les tâches, le protocole
et les budgets. Un verrou empêche deux contrôleurs de la même campagne.
Les workers lourds sont séquentiels et isolés. Le contrôleur mémorise l'identité
PID/date de création de ses descendants ; l'arrêt propre et la reprise ne
ciblent que ses propres processus observés.

Le SHA-256 attendu est
`f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`,
conservé dans les preuves historiques avant/après de H1. Le nouveau hash complet
a également été refusé avant lancement (`hash-admission.guard.json`). Un nom de
blob ou une identité de fichier stable ne sont pas présentés comme une nouvelle
vérification cryptographique des 23 938 321 664 octets du checkpoint.

Les écritures d'état sont atomiques. Chaque expérience achevée est protégée par
des SHA-256 complets du checkpoint avant/après. La reprise conserve les blocs
scellés, vérifie leurs fichiers et relance le bloc incomplet. Les tentatives
refusées ou interrompues restent consultables. Une modification du code,
du protocole ou de l'identité du modèle exige une nouvelle campagne.
Un arrêt brutal au moment exact de la création d'un processus reste une limite
du système ; les baux d'identité persistants réduisent ce risque sans autoriser
la terminaison d'un processus tiers.

Le vérificateur indépendant `benchmarks/verify_m5_validation.py` utilise la
bibliothèque standard Python et le vérificateur de logits hérité. Il contrôle
les fichiers bruts, les commandes, les budgets, les gardes, les arrêts,
les routes, les tokens et les logits FP32 complets enregistrés. Les tolérances
réelles restent `atol=1e-4, rtol=1e-4`. Les rapports miniatures ne remplacent
jamais cette comparaison du vrai checkpoint. La parité complète des tableaux
internes DeltaNet/KV n'est pas certifiée par les seuls logits enregistrés.
Un manifeste garantit la cohérence des preuves, pas leur authenticité absolue.

Les plans couvrent plusieurs paires AB/BA et séparent :

- H1/H2, blocs fixes 16/32/64/128 et contrôle de répétabilité ;
- TensorOps, réutilisation 2/4/8/16, transport groupé ;
- transport historique/direct, import sans copie et pipeline gate/up ;
- spéculation 2/4/8/16 sur copie synthétique, favorable au prompt lookup ;
- génération longue originale, distincte de la copie et du cas historique « 42 » ;
- combinaison conditionnée à la fidélité et au bénéfice réel des composants.

Les kernels doivent d'abord disposer d'une preuve matérielle admissible ; leurs
formes rejetées restent exclues. Une combinaison sans prérequis validés est
exclue explicitement. Les prompts, options et conditions d'arrêt restent figés
dans chaque paire. EOS et plafond de génération sont distingués ; une sortie
trop courte n'est pas appelée décodage soutenu.

Les temps incluent le draft, la vérification, le rejeu, les replis et la
calibration réellement effectués. Les sources enregistrent également les
passages, octets logiques, lectures, coûts spéculatifs, CPU et ressources.
La campagne instrumentée a un coût d'observation ; ses temps ne remplacent pas
une confirmation finale sans observateur. Un processus neuf est distingué d'un
cache OS froid. Aucun cache système n'étant purgé, **le chargement réellement
froid du stockage et la bande passante NAND ne sont pas certifiés**.

## Analyse des lectures et objectifs

Les **178 192** chargements directs et les **135 518 514 304** octets du premier
forward sont des preuves historiques H1/H2, pas des mesures nouvelles.
L'identité `3 × 56 783 + 7 843 = 178 192` relie 95,60 % de ces chargements aux
trois projections d'experts. Le compteur concerne la génération entière ; les
135,52 Go concernent son premier forward. Les périmètres ne se confondent pas.

L'explication et les compromis du transport sont conservés dans
`research/hyperstream3/IO_ANALYSIS_FR.md` : le regroupement peut éviter des
copies, mais contourne le cache individuel et peut augmenter les relectures.
Le nombre d'appels seul ne permet pas de sélectionner le meilleur chemin.

Les **1,584 token/s**, **110,7 s de TTFT**, 364 tokens et 23 passages sont
historiques. À trafic de décodage historique inchangé de 1,964699 Go/token,
3/5/10 tokens/s demanderaient 5,894/9,823/19,647 Go/s de trafic logique.
Ce calcul est conditionnel : il ne mesure ni le SSD actuel, ni un plafond
matériel, ni les gains des nouveaux chemins. Aucun goulot dominant actuel ni
plafond quantitatif complet ne peut être affirmé sans la campagne réelle.

Le disque externe n'est pas connecté. `/Volumes/SSD` est un lien vers le disque
interne. Il reste donc à tester après connexion physique.

## Reproduction et état de livraison

La sélection finale a réellement exécuté **242 tests pytest réussis**, **49
sous-tests réussis** et **64 cas GPU ignorés**, puis **8 tests unittest réussis**
du vérificateur H2. Les 44 nouveaux tests du contrôleur, du banc kernel et du
diagnostic font partie de cette sélection. Elle comprend de vrais processus
légers de contrôle et des tentatives de falsification des preuves ; elle
n'exécute aucun modèle ni calcul GPU. La suite complète Torch/MLX n'a pas été
lancée. Les journaux et le XML sont dans `research/m5_validation/evidence/`.

Le test d'intégration final prépare les 24 expériences et leurs 144 jobs, puis
effectue deux lancements du contrôleur : les deux rendent **75**, avec pression
2, sans aucun enfant de mesure lancé. La seconde tentative reprend le dossier
existant et conserve deux sessions de refus. Le manifeste de cette preuve a
été vérifié après copie ; le vérificateur complet rejette correctement cette
campagne incomplète. Voir `evidence/controller-admission/smoke-summary.json`.

Le protocole détaillé et les commandes autonomes figurent dans
`research/m5_validation/README.md`. Utiliser un environnement disposant de
MLX 0.32.4, NumPy, psutil et pytest. Les contrôles portables n'importent pas
MLX/Torch et ne constituent pas une preuve GPU.

La publication reste en brouillon, dépendante de la PR #12. `main` et les PR
#10/#11/#12 ne sont ni modifiés ni fusionnés. Les preuves de refus ne sont pas
transformées en résultats de performance. La réussite matérielle de la mission
reste à établir lorsque macOS autorisera les workers.
