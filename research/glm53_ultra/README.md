# AirLLM Next Ultra : consolidation GLM M5 et contextes RONIN

Branche expérimentale `research/glm53-m5-ultra-engine`. CPU10/GPU0 reste
le moteur de référence. Les poids GGUF Q2_K_S et les seuils numériques sont
conservés. Aucune activation de production ni fusion dans `main`.

Cette extension transmet au moteur partagé les instructions et le contexte
propres à chaque agent RONIN. Elle conserve les décisions réellement produites
par le backend GLM et prévoit le repli CPU avant activation d'un chemin GPU
refusé. Elle évalue aussi l'opportunité des accélérations structurelles à partir
des trajectoires existantes.

## Reprise du 3 novembre : preuves longues distinctes

La reprise sur secteur apporte une génération CPU avec `mmap_nocache=true`
jusqu'à EOS naturel : 88 IDs, réponse finale « OK », serveur sorti normalement,
gardes inchangés. Ce cas de chat court ne qualifie pas les contextes complets
des600 agents. Son débit serveur vaut0,129552 token/s ; aucun gain causal
contre la tentative de contrôle interrompue n'est annoncé.

La comparaison CPU32 avec cette seule option est bit exacte sur4 956 160
logits, les routes/routeurs/KDA capturés et7 584 écritures d'état observées.
Une sonde distincte répète deux demandes de quatre sorties dans le même contexte,
avec `llama_memory_clear` entre elles : quatre paires de logits et948 écritures
par requête sont identiques, sans croissance du swap observée.

Le candidat tail12/streaming passe aussi sa fidélité32, mais son cycle
chronométré est arrêté sur swap pendant B1 après13 sorties. Le nouveau tail5
est arrêté pendant sa capture après cinq sorties. Aucun des deux ne reçoit
une qualification de vitesse soutenue. Les résultats historiques courts
ci-dessous restent attachés à leurs versions et à leurs périmètres.

Voir le [rapport de reprise et les sept réponses de la mission](REPRISE_2026-11-03.md)
pour les ressources, empreintes, essais interrompus et limites de l'orchestration.

## État des preuves de vitesse

Les PR [17](https://github.com/iliasm-a11y/airllm-next/pull/17),
[18](https://github.com/iliasm-a11y/airllm-next/pull/18) et
[19](https://github.com/iliasm-a11y/airllm-next/pull/19) sont des recherches
empilées. Leurs mesures couvrent des prompts, bibliothèques et longueurs
différents. Elles ne constituent pas un classement universel des backends.

Le dernier cycle V3 clos qui compare les candidats à leur contrôle contemporain
est A/B/B/A, huit sorties par passage, deux passages par bras :

| Candidat V3 | CPU contemporain | Candidat | Variation du décodage |
|---|---:|---:|---:|
| CPU + nocache + anticipation32 | 0,137903 | 0,137456 token/s | −0,324 % |
| Tête Q8 Metal + streaming | 0,141111 | 0,147903 token/s | +4,814 % |
| Tail5 Q8 Metal + streaming | 0,138792 | 0,151471 token/s | +9,136 % |

Le meilleur passage de ce cycle tail5 est 0,153745 token/s. Les deux passages
candidats dépassent leurs deux contrôles. Ces huit sorties terminent sur limite,
sans EOS. Le résultat établit un gain local court ; les seuils 0,5/1/3/5 tokens/s
restent non démontrés. Le préchargement CPU seul n'apporte pas de gain ici.

Les manifestes de fidélité existants V3 et V4 sont conservés séparément,
avec leurs hashes de bibliothèques. Chacun compare 2 478 080 logits sur
16 étapes, bit à bit, avec routes ordonnées, routeurs et KDA capturés. Les
comparaisons d'écritures d'état observées sont conservées dans les manifestes
candidats. Cette preuve limitée à la trajectoire capturée ne valide ni toutes
les écritures d'état possibles, ni les nouvelles conversations des agents,
ni les contextes longs ou l'annulation sous toutes les configurations.
[Manifestes originaux compressés et sélection](evidence/fidelity-manifests/selection.json).

Les chemins V3 utilisent CPU10, batch/microbatch8, contexte512, un slot,
mmap et préchargement des experts déjà sélectionnés. Le candidat tail5 ajoute
les matvecs Q8 compatibles CPU de 35 matrices originales (1 482 883 072 octets),
nocache demandé et anticipation dense32 Mio. Les experts Q2, les normalisations,
le routage et les récurrences KDA restent CPU. Les contributions des réglages
de streaming ne sont pas isolées causalement par ce cycle.

La campagne distincte V4 évalue la rétention par mlock de 103 matrices, soit
268 255 232 octets de pages originales sous budget 256 Mio. Ses hashes natifs et
sa nouvelle fidélité sont évalués séparément. Le cycle V4 est clos :

| Passage V4 | CPU10 témoin A | Tail5 + streaming + cache256 B |
|---|---:|---:|
| Premier passage | 0,088240 | 0,058226 token/s |
| Second passage | 0,030634 | 0,032185 token/s |
| Médiane | 0,059437 | 0,045206 token/s |

Le rapport brut calcule −23,94 % de débit médian dans ce cycle. **La comparaison
de performance est invalidée** : les journaux système enregistrent fermeture
du couvercle et veilles à partir du 3 novembre à 00:12:48, avec des réveils de
maintenance pendant le cycle. Cette observation a été ajoutée après le résumé
initial ; son champ `comparison_valid_on_stated_scope=true` ne vaut donc pas
qualification des conditions d'alimentation. Les originaux restent inchangés.
Voir [observation d'alimentation et veilles](evidence/power-observation.json).

La dérive très forte des deux bras empêche une attribution causale propre au cache. Le dernier
contrôle lit presque le même volume OS que son homologue V3 (116,762 contre
116,759 Go), mais consomme davantage de temps CPU cumulé (1 816,523 contre
436,471 secondes). Une lecture des sources et options principales de compilation
ne trouve pas de changement actif du calcul CPU dans ce contrôle. La cause du
ralentissement reste inconnue ; les fréquences, l'attente active et la charge
concurrente ne sont pas départagées par ces données.

La croissance maximale du swap observée dans V4 vaut 22 740 992 octets, sous
le garde de 32 Mio. Le résumé refuse `optimization_qualified` et
`memory_stability_qualified`. Aucun gain du cache256 n'est donc qualifié.
[Instantané brut clos, compressé sans perte](evidence/cache256-closed/selection.json).

## Chemin critique et possibilités physiques

L'union d'opérandes mesurée sur les étapes capturées est de 12 218 990 592
octets/token, dont 9 444 458 496 denses (77,2933%) et 2 774 532 096 d'experts
sélectionnés (22,7067%). Il s'agit d'accès logiques aux poids originaux.
Les lectures OS du PID et les lectures de blocs du contrôleur entier sont
distinctes ; aucune de ces mesures n'est assimilée à des cellules NAND.

L'ancien diagnostic CPU4 sérialisé répartit 6,2804s/étape en 4,3498s de produits
denses, 1,2348s d'experts et 0,6959s d'autres opérations. Ces coûts incluent
calcul, fautes de pages et ordonnancement. La résidence d'une vraie matrice Q8
réduit son produit de 15,783ms à 0,314–0,349ms dans un essai de composant,
avec disparition des lectures OS observées. Ce résultat ne décompose pas
exactement l'I/O du modèle complet.

Le cache256 actuel a un plafond direct optimiste de +5,709% sur cet ancien
diagnostic, même en supprimant tout le temps des opérations retenues, calcul
compris. Les effets indirects de résidence ne sont pas couverts par cette
borne. Un petit cache ne fournit donc aucune preuve d'une trajectoire vers
une multiplication du débit. Le Fanxiang n'était pas monté et n'a pas été testé.

## Les 600 agents et le moteur partagé

La table historique des600 agents ne suffisait pas à transmettre leur
individualité au modèle. Le nouveau `AgentContexts` stocke maintenant :

- 600 identités stables, chacune avec nom, instructions, rôle et spécialisation ;
- mémoire de travail et historique isolés par paire `(agent_id, mission_id)` ;
- messages système/utilisateur/assistant effectivement transmis au backend ;
- texte GLM, IDs natifs, empreinte du contexte, runtime et provenance disponibles ;
- états `completed` sur EOS explicite, `partial`, `cancelled`, `failed` et
  `interrupted`, sans transformer une sortie limitée en réponse finale.

Un seul `NativeGlmModel` conserve le processus et son slot physique. Les
contextes persistants sont du texte et des métadonnées SQLite ; ils ne sont
pas600 caches KV. Le backend recalcule le prompt avec `cache_prompt=False`.
La persistance et l'isolation de l'hôte ne démontrent pas, seules, la fidélité
numérique de tous les états natifs sur toutes les requêtes.

La mémoire de travail est bornée à 8 Kio, le contexte assemblé à 64 Kio et
l'historique à 128 tentatives par paire agent/mission. La sortie est bornée à
256 Kio et163 840 IDs, avec le budget de tokens natif plus restrictif en pratique.
Une limite produit une erreur explicite. Aucun ancien tour terminé n'est
supprimé et aucun résumé n'est inventé pour faire tenir la demande.

Le tokenizer natif compte le prompt rendu et refuse `CONTEXT_LIMIT` quand
prompt+sortie dépasse le contexte configuré. Le contexte512 historique peut
être insuffisant avec identité, instructions et historique ; tout changement
de contexte exige une configuration explicite et conserve l'admission mémoire.
Le précontrôle offline fourni compte 444 et 443 tokens pour les premières
requêtes des agents 1 et 600, avec une courte mémoire de travail. Ces deux
scénarios tiennent avec 32 sorties dans 512 ; ce n'est pas une garantie pour
l'historique, les mémoires remplies ou le rendu du serveur natif.

Les sorties partielles sont conservées dans l'historique comme brouillons.
Les tentatives annulées, échouées et interrompues restent consultables et sont
exclues de la conversation rejouée, avec cette exclusion indiquée au modèle.
Une génération ne crée aucune action externe à partir du texte du modèle.

À 0,151471 token/s,600 demandes de 32 sorties représenteraient environ 35,2 h
de décodage en calcul conditionnel, avant préfill et orchestration. Il ne
s'agit pas d'une campagne600 agents mesurée ni d'une capacité interactive.

## Repli avant activation

`Engine` refuse un GPU non qualifié avant de construire le backend. Par défaut,
il sélectionne CPU10/GPU0, sans cache GPU, en conservant les limites de ressources
configurées. Le statut expose `engine.backend_selection` et la cause
`GPU_FIDELITY_UNQUALIFIED`. Aucun candidat GPU n'est lancé dans ce cas.

L'option explicite de recherche `allow_unqualified_gpu=true` conserve son
comportement expérimental. `fallback_to_cpu_on_rejection=false` permet de
conserver un refus explicite pour un appel de recherche. Aucune relance
automatique ne suit un début de génération, une erreur mémoire, une annulation
ou un problème d'intégrité : la décision déjà engagée reste enregistrée.

## Intégration et reproduction

Les méthodes stdio existantes acceptent maintenant `agent_id` dans `generate`.
Les méthodes nouvelles sont `agent_identities`, `agent_context`, `agent_memory`
et `agent_decision`. `id` désigne toujours la mission.

```json
{"method":"agent_memory","params":{"id":"MISSION","agent_id":600,"text":"Observation de travail propre à cet agent et cette mission."}}
{"method":"generate","params":{"id":"MISSION","agent_id":600,"prompt":"Analyse les observations disponibles.","max_new_tokens":32}}
{"method":"agent_context","params":{"id":"MISSION","agent_id":600}}
```

Les nouveaux modules sont inclus par le constructeur RONIN existant. Le code
de cette branche n'est pas installé dans l'application en production.

Tests portables, sans preuve d'inférence réelle :

Validation locale : 47 tests RONIN et 17 tests Ultra réussis. Les logs sont
conservés dans `evidence/portable-{ronin,ultra}-tests.log`.

```sh
python3 -m unittest discover -s integrations/ronin/tests -v
python3 -m unittest discover -s research/glm53_ultra -p 'test_*.py' -v
```

Le rejeu structurel se reproduit sans checkpoint :

```sh
python3 -m research.glm53_ultra.structural --bundle research/glm53_ultra/evidence/structural-inputs --output /tmp/glm-structural-result.json
```

Le probe d'intégration réelle crée deux missions isolées, pour les agents 1
et 600, et conserve son serveur partagé. Il exige une configuration CPU10/GPU0
avec runtime et receipt de poids vérifiés, ainsi que le verrou de campagne libre :

```sh
python3 -m research.glm53_ultra.agent_probe --config /CHEMIN/cpu-reference.json --output /CHEMIN/NOUVEAU/agent-probe --tokens 4
```

Il conserve RSS≤ 8,5 Gio et croissance du swap≤ 32 Mio, ou des limites configurées
plus strictes. Il enregistre contexte exact, vrais IDs, nombre de chargements,
PID, gardes et nettoyage. Ce probe ne remplace pas une campagne de débit ni
une comparaison des logits/routes/KDA pour les nouveaux prompts.

Sur Darwin, ce probe refuse aussi un couvercle fermé ou inconnu avant de créer
le runtime. Son lancement réel du 3 novembre est refusé avec
`POWER_CONTEXT_CLOSED`, sans requête native :
[rapport conservé](evidence/agent-probe-closed.json). La nouvelle intégration
des contextes n'a donc pas encore de génération réelle validée. Le secteur est
consigné mais n'est pas imposé. Cette protection du probe n'altère pas
l'admission du backend en production.

Voir [étude des accélérations structurelles](STRUCTURAL.md) et ses petites
entrées reproductibles. La variante GGUF ne contient aucun poids MTP/NextN ;
les opportunités ngram observées ne justifient pas son activation sur cette
charge. Les préfixes et la concurrence supérieure à un slot restent désactivés
dans RONIN jusqu'à qualification des états récurrents et du coût total.
