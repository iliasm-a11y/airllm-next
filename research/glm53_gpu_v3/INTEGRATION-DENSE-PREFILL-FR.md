# Intégration source : poids denses et préfill GLM-5.3 Flash

Base : AirLLM Next `47fb627826822db02132995dbcc1d5a225bb9772`, tête publiée
de la PR #26. Branche locale : `integration/glm53-dense-prefill`.

Les modifications sont intégrées aux modules AirLLM et RONIN de cette branche.
Les options d'accélération restent désactivées par défaut. Aucun test, script
du dépôt, compilateur, benchmark ou modèle n'a été exécuté pour cette
intégration. Le runtime installé, son processus, ses verrous et sa configuration
n'ont pas été modifiés. Aucune activation ou campagne automatique n'est prévue.

## Chemin raccordé

`RONIN Runtime -> Engine._options -> NativeGlmModel -> llama-server`.

Le constructeur conserve ses anciens défauts : contexte 512, quatre threads,
batch 8, GPU0 et aucun nouveau cache. `ubatch_size=None` reprend `batch_size` ;
`threads_batch=None` conserve la sélection native antérieure. La configuration
JSON explicite reste prioritaire sur les variables d'environnement de RONIN.

| Fonction | Paramètres intégrés | Défaut |
|---|---|---|
| Anticipation dense | `dense_lookahead`, `dense_lookahead_mib`, `dense_lookahead_max_cols` | faux, 32 Mio, une colonne |
| Conseil sur la matrice courante | `dense_prefetch_max_cols`, avec `dense_prefetch` | une colonne, désactivé |
| Préfill | `ubatch_size`, `threads_batch` | politique antérieure |
| Préfixe partagé | `shared_prefix_tokens`, `shared_prefix_render_options`, `shared_prefix_scratch_mib` | absent |
| Rétention de pages denses | `dense_lock_names`, `dense_lock_mib` | aucune liste, plafond 128 Mio |

Les limites de lookahead sont 16/32/64/128 Mio et 1/2/4/8 colonnes. Le worker
unique, la tâche unique, les contrôles des poids constants et les admissions
mémoire du prototype sont conservés. Un conseil accepté ne prouve pas une
lecture SSD, une résidence effective ou un chevauchement utile.

Les variables RONIN supplémentaires reconnues sont
`RONIN_GLM_THREADS_BATCH`, `RONIN_GLM_UBATCH`, `RONIN_GLM_DENSE_LOOKAHEAD`,
`RONIN_GLM_DENSE_LOOKAHEAD_MIB`, `RONIN_GLM_DENSE_LOOKAHEAD_MAX_COLS` et
`RONIN_GLM_DENSE_PREFETCH_MAX_COLS`. Les IDs communs et les noms de poids
retenus exigent une configuration JSON explicite. Aucune liste n'est inventée.

## Disponibilité du runtime

Les options denses requièrent le manifeste vérifié du runtime et les patches
correspondants. Une option inconnue du binaire ne doit pas passer silencieusement.
Le lookahead multicolonne exige la capacité `dense_lookahead_multicolumn_v1`.

Le préfixe partagé exige un futur runtime construit avec le paramètre de
recette `prefix_cache_experiment=True`. La recette inclut
`patches/llama-server-prefix-cache.patch`, copie `server-prefix-cache.h` dans
le checkout natif temporaire et déclare `shared_prefix_snapshot_v1` dans son
manifeste. Le binaire actuel n'est pas reconstruit par cette intégration.
Les capacités déclarées ne remplacent pas une preuve de fidélité ou de gain.

Le prototype utilise le serveur existant et un seul snapshot complet plafonné
à 256 Mio. Il vérifie les vrais IDs du préfixe à chaque requête, impose un
suffixe privé, emploie les flags de séquence `NONE` et conserve le sampler
neuf de la requête. Aucun snapshot ne persiste sur disque. Les métadonnées
privées de lancement sont liées aux poids, au runtime, aux paramètres et au
rendu, et ne sont créées que lors d'un futur lancement explicitement demandé.

Ce profil est limité à CPU10/GPU0, batch/ubatch 8, un seul slot, sans LoRA,
multimodal, spéculation ou cache MoE. Il impose RSS <=8,5 Gio et croissance
swap <=32 Mio dans le garde parental, avec une admission initiale >=5,5 Gio.
Le helper réserve le snapshot et le budget explicite de scratch, refuse une
mesure mémoire inconnue et conserve ses contrôles de pression, réserve et swap.
Les caches RAM et checkpoints génériques sont désactivés dans ce profil
uniquement. Un échec de restore ne relance pas la requête ; un échec de decode
ne déclenche pas un nouvel essai avec un batch réduit.

## Frontières et comparaisons

La frontière est issue des IDs natifs complets et alignée sur le microbatch.
Pour un préfixe réellement commun de 211 tokens : 208 pour 8/16, 192 pour 32/64,
128 pour 128. La clé du snapshot lie la frontière, sa politique et les positions.

`prefill_prefix_boundary` peut déclarer une frontière commune aux bras de
recherche, si elle est valide pour leurs vrais IDs et leurs grilles. Les
comparateurs refusent des définitions de préfixe différentes. Les checkpoints
128/256/384/512/final restent requis ; la frontière déclarée s'y ajoute.
Les reçus historiques ne qualifient pas les nouvelles sources ou bibliothèques.

Le microbatch 16 est historiquement rejeté numériquement. Cette intégration ne
prétend pas résoudre cette divergence. Les tailles supérieures restent des
profils de recherche, jamais des accélérations acquises.

## Vérification réalisée et qualification restante

Réalisé : correspondance de la base Git, application du correctif sans conflit,
relecture statique, contrôle des chemins et des options. Les régressions de
configuration, fermeture des flux, frontière et identité sont intégrées aux
suites existantes. Aucun test n'a été exécuté et aucun binaire n'a été construit.

La qualification ultérieure devra établir séparément la compilation, les
refus de ressources, l'annulation, les logits complets, routes ordonnées,
états KDA/attention/indexer et l'isolation1->600->1. Le temps net doit inclure
création, garde, hash, reset, restore, suffixe et réponse utile jusqu'à EOS,
avec un témoin comparable et une même origine de swap. Aucun facteur de
vitesse n'est annoncé avant ces mesures.
