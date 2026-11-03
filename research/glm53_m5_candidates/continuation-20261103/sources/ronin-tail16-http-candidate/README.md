# Candidat HTTP 16 — source isolée, non exécutée

Cette copie séparée prépare le portage de la politique native
`cpu8-tail-compatible-v1` vers RONIN. Le répertoire HTTP 8 d’origine, son manifeste
et ses 44 tests restent inchangés. Aucun serveur, modèle, build ou GPU n’a été
lancé pour cette copie. Les valeurs de performance, d’EOS et de fidélité HTTP
restent non mesurées. Le mode par défaut du runner affiche seulement son plan.

Deux admissions différentes sont implémentées :

| Mode explicite | Preuves natives réellement exigées avant construction du Runtime | Usage des slots |
|---|---|---|
| Sans cache commun, par défaut | Pont CPU8 → full16 v1 **complet**, exact, gardé, lié aux vrais builds/sources/patch par le comparateur épinglé `07939855…` | Erase avant/après chaque requête ; aucune sauvegarde/restauration de préfixe commun |
| Avec cache commun, `common_cache=True` | Même pont complet, **plus** full16 → restore16 strictement positif avec le payload canonique à 208 | Quarantaine complète du préfixe, validation SHA/IDs/taille, restauration et resauvegarde contrôlées avant chaque suffixe |

Une restauration native arrêtée pour swap ne bloque donc pas, à elle seule,
le mode sans cache commun. Elle bloque le mode commun. Une première requête
exacte, un bras partiel, un ancien CPU16 divergent, des déclarations JSON ou
les mocks locaux ne satisfont aucun de ces gates. Le mode sans commun refuse
les chemins de restauration fournis par erreur au lieu de les ignorer.
La v2 à fences 208 est un autre contrat, non accepté par ce pin v1.

Le checkout reste `work/airllm-next`, lié au comparateur v1 frère. Le superviseur
UID existant est chargé séparément depuis `work/uid_supervisor.py`, avec son SHA
`7857d7d0…c219` et son `REPO` vérifiés. Les sources/receipts/binaries mesurés ne
sont ni copiés sur eux-mêmes ni modifiés. `guards.py` est la copie byte à byte
de la garde HTTP 8 déjà revue : preload honnête avec probe libéré, puis vrai FD
natif retenu/hérité et identité de l’enfant ; PowerGate et NativeLimits restent
les gardes existantes. Aucun deuxième flock n’encadre le chargement natif.

## Découpage des queues impaires

Le serveur existant conserve son batching ordinaire 16. La politique corrigée
du helper est reproduite par deux tâches HTTP uniquement quand le dernier lot
privé est impair et contient plus de huit tokens :

| Prompt complet | Mode sans commun | Mode commun admis |
|---:|---|---|
| 520 IDs | Génération directe, `cache_n=0`, prompt traité 520 | Restore 208, génération directe, `cache_n=208`, suffixe 312 |
| 521 IDs | Préparer 520 depuis 0, puis envoyer les 521, `cache_n=520`, suffixe 1 | Restore 208, préparer 520 depuis 208, puis les 521, suffixe 1 |
| 523 IDs | Préparer 520, puis les 523, suffixe 3 | Restore 208, préparer 520, puis les 523, suffixe 3 |
| 527 IDs | Préparer 520, puis les 527, suffixe 7 | Restore 208, préparer 520, puis les 527, suffixe 7 |
| 529 IDs | Génération directe, `cache_n=0` | Restore 208, génération directe, `cache_n=208` |
| 542 IDs | Génération directe : queue paire de 14 | Restore 208, génération directe : queue paire de 14 |

La préparation utilise honnêtement `n_predict=0` et exige **un** échantillon
effectivement produit, compté et écarté. Ses IDs/timings, `tokens_predicted=1`,
`tokens_evaluated=prepare_end`, `cache_n`, `prompt_n` et le slot idle à cette
frontière doivent concorder. Le sample ne devient ni un message RONIN ni un ID
ajouté au prompt. La seconde tâche reçoit toujours le **prompt complet**, pour
que le serveur réutilise son préfixe ; elle doit déclarer exactement la frontière
attendue et son suffixe non vide. Une queue de 11 devient 8 + 3, jamais 10 + 1.
Le test de découpage couvre les longueurs légales 209 à 1 536, depuis 0 et 208.
Ce test de contrôle de flux ne prouve pas le calcul natif réel.

Sans cache commun, cette réutilisation est seulement l’état privé transitoire
de la même requête logique. Aucun état d’une identité précédente n’est réutilisé.
L’erase est observé avant et après chaque contexte. Il s’agit de l’effacement
de la séquence valide exposé par V4 ; aucun effacement sécurisé octet par octet
de toute la RAM du serveur n’est revendiqué.

## Budgets, profil et runner

Les paramètres demandés sont CPU 10, GPU 0, un serveur/slot, batch et microbatch
16, contexte 2 048, swa_full, K/V f16, sans offload KV/op ni repack/extra buft,
sans cache RAM/checkpoints/draft. Les variables de capture natives restent dans
la preuve, sans passer au serveur. Les presets automatiques sont isolés/refusés
comme dans la copie HTTP 8. Les paramètres non exposés par `/props` restent
une cible déduite des sources, pas un descripteur effectif HTTP mesuré.

Une clé HTTP propre lie les sources de cette copie, le vrai serveur chargé,
le manifeste natif admis, l’environnement, les flags, les limites, le plan de
queue et le choix avec/sans commun. Les deux modes ont des clés différentes ;
aucune clé CPU native ou HTTP 8 ne leur est attribuée par normalisation.

Le runner valide la configuration CPU 8 de référence avec son parseur durci,
puis journalise le seul changement de profil `batch_size: 8 → 16` et fournit
la configuration résultante à sa factory isolée. Il conserve les 600 identités,
les missions privées et le parcours 1 → 600 → 1 avec l’historique stocké complet
de l’agent 1. La garde du modèle, son verrou, son nettoyage et son sampler natif
restent hérités. La factory de production n’est pas modifiée.

`logical_scope()` commence avant `Runtime.dispatch`. Les deux tâches privées,
rendu/tokenisation, erase/restauration/resauvegarde, attente et vraie génération
partagent **un** délai de 2 700 s. Le budget global de 7 200 s inclut la relecture
des preuves, le chargement et la préparation commune éventuelle. Les 512 tokens
de sortie utile restent le budget de la vraie génération ; un sample privé
écarté est compté séparément, donc jusqu’à 513 samples physiques pour cette
requête logique. Aucun coût de préparation n’est exclu du délai.

Le journal physique horodaté conserve tous les appels HTTP et les vrais événements
SSE. Le TTFT logique inclut les étapes privées ; le TTFT du flux natif est séparé.
Tout échec de cache, slot, format, sample, garde, timeout, annulation ou abandon
ferme l’enfant détenu et interdit la reprise automatique. Les erreurs natives
ne sont pas changées en succès, EOS ou tentative avec moins de contrôles.
Les deux premiers EOS utiles éventuellement observés sont préservés séparément
si le troisième tour échoue ; ils ne qualifient pas toute la campagne.

## Limites de fidélité et vérification future

Le pont natif est une admission de recherche, pas une preuve numérique HTTP.
Les contrôles de frontière/cache/tokens sont nécessaires et ne remplacent pas
les 154 880 logits F32, routes ordonnées, routeurs, sorties KDA et états complets.
Même une future campagne C utile laisse `HTTP_numerically_qualified`,
`exact_state_qualified` et `full_qualification` à faux avec ce serveur actuel.

`http-verification-protocol.md` décrit les passages diagnostiques à comparer
aux fichiers natifs gelés et les blocages d’API : probabilités/logprobas ne sont
pas des logits absolus, et les routes/KDA ne sont pas exportées. Les sauvegardes
N/N+3 demandent un harness diagnostique distinct avant erase et un parseur
généralisé ; ils ne sont ni exécutés ni déclarés disponibles ici. Une future
instrumentation devra avoir son propre build/profil et de nouveaux témoins.

Les tests positifs de cette copie remplacent explicitement les gates natifs,
la liaison du package et le serveur. Leur état synthétique ne fait que quelques
dizaines d’octets. Ils n’activent ni ne qualifient aucun modèle ou cache.
