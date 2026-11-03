# Localité mmap : option explicite, non qualifiée pour les agents longs

La [reprise du 3 novembre](REPRISE_2026-11-03.md) clôt la parité CPU32, un chat
court jusqu'à EOS88 et un reset natif2×4 sur cette option. Ces cas ne constituent
pas une qualification générale des longs contextes d'agents ni une preuve
d'accélération. Le défaut reste `false` ; les résultats et échecs du protocole
ci-dessous sont conservés séparément dans le rapport.

`NativeGlmModel` accepte maintenant `mmap_nocache`, booléen strict avec défaut
`false`. Le statut indique la valeur demandée dans `configuration.mmap_nocache`.
Le JSON explicite prévaut sur `RONIN_GLM_MMAP_NOCACHE=1`, selon le contrat des
autres options booléennes de l'adaptateur. Tous les `AIRLLM_*` hérités sont
nettoyés; seul le choix explicite réintroduit `AIRLLM_MMAP_NOCACHE=1` dans
l'environnement du serveur détenu.

Dans la source native expérimentale Apple existante, cette variable ajoute
`MAP_NOCACHE` à `MAP_SHARED` avant le mapping en lecture seule. Le journal
natif annonce `low-locality mapping hint enabled`. C'est un conseil de
localité au système; cela ne prouve ni éviction immédiate de toutes les pages,
ni lecture physique du SSD à chaque accès, ni suppression du cache système.
Un runtime ne contenant pas ce patch peut ignorer la variable : la valeur du
statut Python décrit la demande, et les logs/manifestes doivent confirmer le
chemin réellement présent dans le binaire mesuré.

Les octets des poids, la quantification, les routes et les opérations
arithmétiques ne sont pas modifiés par cette option. Ce constat sur le code
n'est pas une qualification numérique d'une nouvelle compilation ou d'un
nouveau prompt. Les essais courts V3/V4 sur le prompt de six tokens ne
valident pas automatiquement le préfill des agents d'environ444 tokens.

L'hypothèse à tester est qu'un conseil de faible localité pourrait réduire la
conservation de pages de poids au détriment d'autres pages système pendant un
préfill long. Il peut aussi augmenter les relectures et ralentir le calcul.
La croissance globale du swap ne suffit pas à attribuer causalement les pages
échangées au processus modèle. Aucune amélioration mémoire ou vitesse n'est
annoncée par l'ajout de cette option.

## Qualification proposée après les générations longues prioritaires

- Conserver CPU10/GPU0, un slot, contexte512 et batch/ubatch8, même runtime,
  reçus, paramètres de template et prompts exacts. Garder toutes les limites
  RSS/swap/pression/admission/durée inchangées.
- Ne changer que `mmap_nocache=false/true`. Ne pas combiner d'abord ce choix
  avec un changement de checkpoints, cache, lookahead, concurrence ou kernels.
- Vérifier le journal natif d'activation et le manifeste du binaire. Comparer
  logits, routes ordonnées, routeurs et états KDA selon les critères existants,
  puis effectuer des répétitions alternées et rapporter les échecs aussi.
- Séparer durée de chargement, préfill/TTFT, décodage et durée totale; conserver
  RSS/phys_footprint, mémoire disponible, pression, swap et lectures observables
  en distinguant leurs périmètres. Aucune lecture logique n'est une mesure NAND.
- Rejeter le candidat au moindre écart ou arrêt de garde. Le défaut reste
  `false`; aucune reprise automatique d'une décision engagée ni assouplissement
  des protections n'est ajouté.

Les demandes longues des agents reprennent après la priorité de qualification
32 tokens/EOS de référence. Le code de configuration seul ne rend pas ce
nouveau chemin intégrable en production.

## Observation conservée de la fin du processus natif

`status.last_owned_process_exit` conserve le PID détenu, sa génération, le code
de sortie retourné par `Popen.poll()`, l'instant d'observation, le chemin de
nettoyage et l'éventuel code de garde. `exit_observed=true` exige un vrai code
de sortie entier. Une référence de processus devenue `None`, une erreur de
lecture ou un code absent ne constituent pas une preuve d'arrêt.

Cette observation survit au nettoyage répété et est remise à zéro lors du
prochain lancement détenu; une ancienne génération de watchdog ne peut pas
écraser celle du nouveau processus. Le probe utilise cette preuve lorsque la
garde a déjà fermé le serveur avant son propre nettoyage. Observer la fin du
serveur ne transforme jamais une inférence échouée en réussite. Les signaux,
temps d'attente, seuils de garde et décisions de terminaison restent inchangés.
