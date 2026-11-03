# noBLAS + NoCopy, sonde native de rejet à 128 tokens

Statut : source expérimentale préparée, non compilée/non exécutée, désactivée par défaut. Aucun nouveau
helper n'est construit, aucune nouvelle mesure GPU/modèle n'a été réalisée par
cette préparation. Les bibliothèques et les preuves NoCopy closes sont en lecture
seule. Les sources v3 ne sont **pas** qualifiées sur le prompt complet : la queue
30 a révélé un autre chemin CPU. La géométrie v4 a été copiée depuis le source gelé c37423f3444ec5f986515e3e2c31abba32a2d43967eb0e49bde4290335bc3a78 ; aucune qualification modèle v4 n’est héritée.

## Raccordement concret et réutilisation des bibliothèques

Le provider compilé existant se trouve dans
`work/metal-tensorops-q8-nocopy/build-run1/runtime/bin`. Son reçu réel
`provider-build.json` a pour SHA256
`4fe68689c17b1c335f7b2fab2087aa99cf179d1dbe760d2a810d9cec1b01cb63`.
Il inclut le provider CPU qui quantifie les activations avec la routine native,
calcule les dots INT8 par blocs32 sur Metal et emprunte une vue bornée des poids
Q8 originaux. Le CPU_Mapped avec usage WEIGHTS est accepté explicitement. Le
provider n'effectue aucune copie permanente de l'ensemble des poids.

Le SHA de profil est lu dans l'environnement à l'initialisation du provider
(`metal-q8-bridge.mm`, `hook_provider`), et n'est pas un SHA de profil figé dans la
dylib. Un nouveau domaine de profil peut donc réutiliser ces **mêmes octets** de
bibliothèques : le kernel, le patch, les six bibliothèques, le reçu provider,
le driver, ses deux dépendances, le contrôleur, son contrat et le garde original
sont identifiés séparément. Cette possibilité de liaison n'est pas une preuve
d'exécution du nouveau helper.

Le nouveau `build_driver.py` ne configure ni ne reconstruit les mathlibs. Il
reprend la commande réellement enregistrée de l'ancien driver NoCopy, en changeant
seulement la traduction C++ et la sortie. Les headers viennent de la source
réellement compilée NoCopy ; remplacer ces headers par ceux du runtime original
serait incorrect, car `ggml-backend.h` contient aussi l'API additive du provider.
Le nouveau reçu driver conserve les SHA avant/après de toutes ses entrées, le
reçu provider inchangé, les stdout natifs réels des paramètres UB8..128 et du
registre, ainsi que le binaire signé réellement obtenu. Aucun reçu `built` ni
résultat de smoke n'est fabriqué dans cette préparation.

## Pourquoi retirer BLAS et conserver les fences/queues

Le mécanisme démontré séparément est celui du scheduler original : les poids Q8
CPU/host deviennent admissibles au backend ACCEL BLAS à partir de N32, même avec
`op_offload=false`. Ce calcul déquantifie les poids vers F32 et multiplie les
activations F32 ; il peut donc différer du dot CPU Q8 natif et éviter entièrement
le hook CPU NoCopy. Le nouveau helper appelle l'API publique
`ggml_backend_unload(BLAS)` après son unique `ggml_backend_load_all`, avant le
modèle. Il exige CPU et zéro ACCEL, et recontrôle le registre après
`llama_backend_init`, après le modèle, après le contexte et avant sortie.

Les descriptions de paramètres et de dyld ne rappellent jamais `load_all` : elles
ne doivent pas réintroduire BLAS. Le chargement natif conserve GPU0 pour le
placement des poids, `devices={nullptr}`, mmap et `use_extra_bufts=false`, sans
offload KQV ni cache MoE. Tout usage GPU effectif est néanmoins nommé explicitement
`native_cpu_with_metal_q8_bridge` dans le profil, la clé et la trace. GPU0 ne
signifie donc pas absence de calcul GPU.

Les fences absolues 128/208/256/384/512 et les queues CPU8 sont conservées dans le
contrat. La correction source v4 découpe aussi les intérieurs
non multiples16, pour éviter le dr1 impair du dispatch CPU sur la queue30. Cette
sonde128 ne vérifie pas cette queue ni la fin du prompt : cette limite reste
explicite même après le raccordement à la source v4.

## Contrat strict de la sonde

L'entrée est le JSON complet gelé `work/native-prompts-max/native-prompts.json` ;
aucun agent ni token ID n'est remplacé, tronqué, re-rendu ou re-tokenisé. Le
contrôleur ajoute seulement la nouvelle clé et la borne de scratch dans un fichier
distinct qu'il suit ensuite par identité. Le helper conserve ses trois séquences
complètes 1→600→1, mais évalue exclusivement request0 positions0..127. Les tailles
et le SHA de la **séquence complète** de request0 sont conservés dans la preuve.

Tous les flags logits des lots sont zéro : la borne128 n'est pas la fin du prompt.
Aucun `llama_get_logits`, sampler ni continuation n'est appelé. Le rapport final
est `status=probe_only`, `complete=false`, `full_prompt_computed=false`,
`full_campaign_complete=false`, `whole_model_qualified=false`. La fin de la seule
fenêtre est indiquée par `diagnostic_window_complete=true` et n'est pas un succès
du prompt complet ou d'un benchmark de préfill521.

L'ancien comparateur complet à trois requêtes ne peut pas être réutilisé en
ignorant son rejet `complete=false`. Le futur comparateur de rejet doit lier les
deux vrais builds et leurs propres clés/gardes, puis comparer la couverture exacte
des 118 séries attendues de request0 jusqu'à128, avec métadonnées, axes, strides,
SHA/bruts et finitude. La barrière de rejet128 proposée compare les118 séries intégralement bit à bit.
Les tolérances Metal originales (atol1e-3/rtol1e-4) restent un diagnostic séparé
et ne convertissent pas un écart du contrat exact en réussite. Routes, états opaques et autres
contrats exacts restent exacts ; aucune tolérance ne doit être inventée pour
faire passer une divergence. Même sans écart observé à128, aucune qualification
du prompt complet, EOS, cache ou performance n'en découle.

## Budgets et limites réelles

- Hard caps originaux : RSS8,5GiB, croissance de swap32MiB, secteur/lid/horloge,
  réserve768MiB et warning headroom512MiB, grâces3s/20s. Aucun seuil n'est modifié.
- Admission initiale pression1 ; pendant le calcul le provider conserve son
  sous-ensemble conservateur pression1/2, avec réserve additionnelle128MiB,
  warning headroom et même baseline swap que le watchdog externe autoritaire.
- Trois buffers Metal fixes : 4 751 360 octets de payload logique. Une seule
  vue temporaire de poids au plus35 782 656 octets (incluant marge d'alignement),
  poids utiles au plus35 651 584 octets. Aucun payload logique n'est présenté
  comme un pic RSS, une résidence pilote ou un coût pinned mesuré.
- `MTLDevice.currentAllocatedSize` est mesuré séparément par le provider, avec
  un plafond64MiB. Graph workspace + payload fixe + vue empruntée doivent aussi
  rester sous64MiB. Le workspace GGML réel peut être supérieur à celui d'une
  fixture, et provoquer un refus pré-submit ; il ne faut pas contourner ce refus.
- Le contrôleur réserve séparément un diagnostic d'état256MiB et scratch16MiB ;
  le checkpoint128 vérifie la RSS avant cette allocation et son scratch.
- Un seul enfant natif, BENCHMARK_LOCK et le vrai verrou UID du modèle détenus
  sans attente. FD UID hérité observé avant GO ; le pipe GO bloque le helper avant
  l'initialisation explicite des backends/modèle. Arrêt par PID/birth/exe propre,
  pas de recherche/kill de processus étrangers. Limite globale proposée1800s,
  échantillonnage0,5s. Durée réelle du probe, RSS/pilote et coût SSD : inconnus.

Le GPU n'est admis comme **observé** que si des commandes multi-token du N demandé
sont réellement soumises, terminées, publiées et appariées à la même forme/poids,
avec identité canonique et kernel, zéro copie des poids, pointeur alias vérifié,
allocations et gardes valides. Activer/importer un profil ne satisfait pas cela.
Une erreur après submit demeure fatale, sans fallback CPU pendant une écriture
GPU possible ; le provider compilé est inchangé.

Les fixtures warm résidentes et l'ABBA interrompu ne démontrent ni le comportement
GLM out-of-core, ni les octets physiques SSD, ni un gain global. Cette sonde est
une nouvelle hypothèse mesurable ; aucun second moteur ni rerun ABBA n'est requis.

## Avant tout lancement

1. Relire la provenance source v4 et le diff helper/contrôleur/contrat.
2. AST/JSON et13 tests portables vérifiés, sans bibliothèques natives ; refaire seulement si un correctif le justifie.
3. Dans une fenêtre zéro modèle explicitement coordonnée : compiler le seul
   driver, obtenir ses smokes/rpaths/dyld réels, puis son binding réel au provider.
4. Faire relire le comparateur de rejet128, qui reste distinct d'une qualification.
5. Seulement après un signal root et admission intacte : une sonde N32/request0,
   sans autre modèle/compilation/charge GPU. Refus et erreurs sont conservés.

## Reçu de préparation

Le plan source `driver-build-planned.json` a été produit par le chemin par défaut
(sans `--execute`) : `planned_not_built`, `build_attempted=false`, aucun binaire
nouveau et aucun stdout natif inventé. Il vérifie en lecture les inputs du reçu
NoCopy existant, ses six bibliothèques et la source v4 gelée. Les13 tests portables
ont passé en0,002s ; aucun test ne compile ni ne charge une bibliothèque native.

## Correctifs de revue source (CPU v4 actif)

Les deux P2 de la revue indépendante sont corrigés en source : le lanceur exige
les600 IDs entiers uniques1..600 et le SHA exact du JSON gelé complet. Il lit une
fois les mêmes octets bornés pour les hacher puis les parser. Aucun champ complete
absent de l'entrée originale n'est inventé. Le chargement de gpu_contract et du
runner appelé par le builder est exact par chemin et SHA ; un module générique
en cache d'un autre dossier est refusé avant tout build ou appel natif.

Le nouveau comparateur est décrit dans PROBE-COMPARATOR.md. Son validateur de
garde partiel est distinct : aucun statut de reçu n'est remplacé pour appeler un
comparateur complet. Les 118 séries brutes, les propres clés, les deux vrais
builds et les gardes closes sont obligatoires ; tous les résultats restent
complete=false/fullqualified=false. Les bits CPU, l'état opaque à128 et les
tolérances Metal auxiliaires sont rapportés séparément.

Aucun test/AST/build/natif/GPU n'a été exécuté pour cette nouvelle révision pendant
le modèle CPU v4. Les13 tests antérieurs appartiennent à la révision précédente.
Le plan de driver précédent est marqué obsolète ; aucun binaire/reçu de build
nouveau ni résultat numérique n'est créé par ces corrections.
