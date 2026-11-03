# Serveur GLM CPU sans BLAS — candidat source uniquement

Le patch est OFF par défaut. Il ne modifie aucune bibliothèque mathématique,
aucun helper ou serveur mesuré et aucun checkout V4 existant. Il n'a été ni
compilé ni lancé. Aucune preuve numérique, de cache, d'EOS ou de performance
HTTP n'est attribuée à cette nouvelle source.

## Insertion réelle

`tools/server/main.cpp` est seulement un wrapper. Le chemin utile est
`llama_server(argc,argv)` dans `tools/server/server.cpp`, après le succès de
`common_params_parse` et avant `llama_backend_init`, `server_run` et toute
création de modèle/contexte. Le parser charge des backends dans ses callbacks
devices, override de buffers et RPC (`common/arg.cpp:229/1115/1132/1158`). Un
unload dans main avant le parser serait donc insuffisant.

Dans la seule branche activée, le patch effectue un dernier `ggml_backend_load_all`,
puis appelle l'API publique `ggml_backend_unload(regBLAS)`. Il exige ensuite
le registre et un device CPU, l'absence de BLAS et zéro device ACCEL. Le CPU
restant empêche le load_all conditionnel de `llama_backend_init` (llama.cpp:132).
Les chemins courants common/server ne rappellent pas load_all après ce point.
Le patch vérifie néanmoins à nouveau registre et configuration avant chaque
load/reload de modèle, puis après `common_init_from_params`.

L'entrée embarquée `llama_server(common_params&,...)` refuse explicitement le
mode1 : un appelant peut déjà retenir un contexte/backend. Aucun registre
process-global n'y est retiré à l'aveugle. Le mode0 garde son flux antérieur.

## Domaine admis et vrais métadonnées

`AIRLLM_GLM53_CPU_NO_BLAS=1` est la seule activation ; absent/0 conserve le
comportement précédent. Les autres valeurs sont refusées avant modèle.
La valeur est immuable durant le processus. Le mode exige les paramètres CPU
explicites déjà utilisés par le runner RONIN natif : gpu_layers0, device none,
no-op-offload, no-repack, fit off, moe_cache0, sans split/override de buffers.
Il accepte seulement un fichier local ; router, worker géré par router,
HF/URL/docker, draft/spéculation, mmproj et adaptateurs restent hors du candidat.

Le nom d'un dossier n'est jamais une preuve d'architecture. Le lecteur GGUF
public est utilisé avec no_alloc=true et ctx=nullptr : seuls métadonnées et
descripteurs sont lus, sans poids ni modèle. Un callback borne les lectures
à64Mio ; cette borne n'est pas une mesure/plafond de heap du parser. Les gardes
RSS/admission existantes et les SHA originaux des shards restent requis par
le propriétaire externe. Le fichier doit déclarer réellement glm5-next et les
comptes du GLM ciblé :45layers,288experts,8sélectionnés,4hyperconnexions,
deuxshards et premier shard0. Types entiers non signés vérifiés avant getters.
La taille, mtime et cible canonique doivent rester identiques jusqu'au load.
Ces contrôles ne remplacent pas la vérification intégrale des poids originaux.

Les preuves conditionnelles vont dans les logs sous
`AIRLLM_GLM53_CPU_NO_BLAS`, et dans `/props.experimental_native_backend_policy`.
Elles exposent politique versionnée, registres avant/après, CPU présent,
ACCEL0, gpu_layers0 et portée du contrôle. Les qualifications géométrie,
numérique du modèle, cache et performance sont explicitement fausses. Il
s'agit de preuves de configuration/registre, pas d'une trace de chaque op.

## Défaut et refus à vérifier plus tard

- Sans variable ou avec0 : aucune lecture GGUF supplémentaire, unload, mutation
  de paramètres ou propriété supplémentaire. BLAS reste disponible normalement.
- Avec1 et configuration admise : retrait après parser, preuves, puis load.
- Avec une valeur telle que true, avec GPU/auto, mauvais GGUF, deuxième shard,
  registre CPU absent ou ACCEL restant : retour nonzero avant modèle.
- Le mode1 refuse l'entrée embarquée et les workers gérés par router.
- Un rechargement ultérieur de BLAS ou un changement du fichier/configuration
  fait refuser load/reload ; la preuve d'activation ne suffit pas à l'ignorer.

`--help` termine dans le parser avant cette politique : il ne peut pas servir
à démontrer une activation ou un refus. Aucun de ces scénarios natifs n'a été
exécuté. La recette et les contrôles négatifs restent un plan.

## Build minimal prévu

Le plan utilise une nouvelle source reconstruite depuis le commit V4 épinglé
et ses16 patches exacts, puis ce patch serveur de trois fichiers. Il configure
les mêmes options/compiler/SDK V4 et demande seulement la cible llama-server,
avec jobs2 sous le superviseur existant. Un build frais peut compiler ses
dépendances ; aucune durée n'est prédite et rien n'est exécuté maintenant.

Le package de revue est neuf. Seuls le nouvel exécutable llama-server et le
nouveau libllama-server-impl.dylib y sont pris du build. Tous les autres
artifacts/bibliothèques sont copiés du runtime V4 original et doivent garder
leurs SHA exacts, notamment llama, ggml/base/cpu/blas/metal, llama-common et
mtmd. Les dépendances et rpaths du nouveau serveur doivent être vérifiés contre
ces fichiers avant tout smoke. Les mathlibs produites transitoirement par
CMake ne sont ni substituées au témoin ni utilisées comme preuve de fidélité.
L'ABI reste celle des en-têtes V4 identiques : aucun struct public n'est changé.

## Blocages avant CMax

La politique ne modifie pas le découpage des tokens du serveur. Elle ne crée
ni fences128/208/256/384/512 ni traitement CPU8 des queues dans HTTP. Même si le
helper noBLAS32 complet réussit, le protocole serveur doit encore démontrer ces
frontières, la fidélité routes/logits/états et le contrat de deux agents.

Le runner de production retire les variables AIRLLM dans son environnement
(`glm_native.py:_runtime_environment`). Un export shell ne suffit donc pas à
activer cette variante via RONIN. Un futur runner isolé devra autoriser
explicitement cette seule variable et inscrire politique/source/binaire/libs
et paramètres effectifs dans une nouvelle identité. Le runner actif est intact.
