# Préfill par lots et snapshot GLM sur M5

Cette branche de recherche part de PR #22 au commit
`64d01af445b620253209319d43a812d1ed07cd13`. Elle conserve le checkpoint
GLM-5.3 Flash Q2_K_S original, les 600 identités logiques, un seul slot et
les protections existantes. Les objectifs GPU, cache exact et deux agents
jusqu'à EOS sont évalués séparément. Aucun chemin nouveau n'est activé dans
RONIN en production et aucune fusion dans `main` n'est demandée.

## Implémentations

- `metal_q8.metal` exécute un produit matriciel réellement multi-token avec
  réutilisation de tuiles. `metal_probe.py` le compare à la routine ARM puis
  au dispatcher GGML CPU10. [Audit Metal](metal_audit.md).
- `snapshot.h` stocke une seule séquence complète avec les flags natifs
  `NONE`, sous plafond de 256 Mio, empreinte et clé de provenance. Il refuse
  les états partiels, les clés incompatibles et les suffixes vides. Les
  erreurs invalident le snapshot et vident le contexte.
- `native_prefill.cpp` offre trois diagnostics CPU : calcul complet,
  découpage explicite à 208, puis préfixe calculé une fois et restauré pour
  `1 → 600 → 1`. Les matrices et tailles de microbatch sont observées dans
  `operations.jsonl`. Les captures numériques sont distinctes des mesures
  de vitesse. [Audit des états](snapshot_audit.md).
- `native_compare.py` exige des captures et des reçus valides avant de
  comparer tous les logits finis, routes ordonnées, routeurs, sorties KDA
  et séquences finales sérialisées. Quatre tokens natifs greedy, leurs logits,
  captures et post-états testent aussi la continuité. Ils ne qualifient pas
  une réponse utile jusqu’à EOS. Il ne transforme pas un essai incomplet
  en comparaison favorable.
- `campaign.py` réutilise le cycle surveillé PR #22, expose séparément batch
  logique et microbatch, l'admission adaptative, les prompts complets et la
  séquence de missions `1 → 600 → 1`. Le retour à l'agent 1 conserve sa
  mission et son historique. Les missions de recherche sont neuves et
  isolées ; elles n'importent pas d'historique personnel de l'application.
- `prompts.py` reconstruit les 600 prompts complets et applique le template
  et le tokenizer natifs sans forcer l'effort de raisonnement.

## Contrat et résultats établis

L'effort habituel du template GLM est **Max** lorsqu'aucun argument n'est
fourni. Le pilote PR #22 imposait `low` ; ses temps ne sont donc pas des
témoins directs de la présente campagne native Max.

La nouvelle tokenisation native des 600 identités confirme **211 tokens
communs** et une frontière candidate de **208**. Les deux nouveaux prompts
complets comptent **521 / 542 tokens**. L'alignement ne prouve aucune
restauration exacte. Le calcul du préfixe est encore nécessaire pour chaque
agent tant que le diagnostic complet ne passe pas.

Les fixtures initiales reproduisent **13 888 valeurs** bit à bit contre la
routine ARM `nrc=2`. Le dispatcher CPU10 change de chemin selon la forme ;
certaines petites formes en lots 64/128 présentent des différences. Ces
observations historiques restent distinctes de la campagne représentative
mesurée ci-dessous. Aucun de ces composants ne qualifie le graphe GLM complet.

La sérialisation de séquence couvre la mémoire hybride, mais **pas les
logits ni le sampler**. Le snapshot est limité à la fin d'un préfixe avant
génération ; un suffixe privé non vide et un sampler neuf sont obligatoires.
Il ne sert pas de sauvegarde universelle d'une génération interrompue.
Le budget temporaire de sérialisation de 16 Mio est une enveloppe prudente
d'admission, pas une mesure de son pic réel. Le budget global charge aussi
le snapshot, les états actifs et les captures temporaires éventuelles.

Les campagnes closes sont indexées dans `evidence/` ; le verdict final reste
distinct des observations de diagnostic.
Les champs non mesurés restent `null`. Les 937,169 secondes historiques et
les essais GPU des PR #17 à #21 gardent leur périmètre d'origine.

## Reproduction et intégration de recherche

Le modèle reste externe. Utiliser une configuration CPU10/GPU0 avec reçu SHA
des shards inchangés et manifeste du runtime épinglé à
`3f8af15a76ac8675812e856da45454bafc90f12c` plus les 16 patches V4. Les sources
des patches et les empreintes des bibliothèques doivent correspondre au
binaire employé. Le helper nouveau doit être recompilé après modification.

```sh
# Répertoire racine du dépôt ; PYTHON dispose de psutil/numpy.
$PYTHON -m research.glm53_batched_prefill.campaign \
  --config /absolute/research-config.json --output /absolute/new-plan \
  --mode plan --microbatch 128

$PYTHON -m research.glm53_batched_prefill.prompts \
  --config /absolute/research-config.json --source /absolute/llama-source \
  --output /absolute/new-prompts

clang++ -std=c++17 -O2 -Wno-deprecated-declarations \
  -I /absolute/llama-source/include -I /absolute/llama-source/ggml/include \
  -I /absolute/llama-source/vendor \
  research/glm53_batched_prefill/native_prefill.cpp \
  -L /absolute/runtime/bin -Wl,-rpath,/absolute/runtime/bin \
  -lllama -lggml -lggml-base -o /absolute/native-prefill

$PYTHON -m research.glm53_batched_prefill.uid_supervisor \
  --wrapper research/glm53_batched_prefill/native_run.py \
  --receipt /absolute/new-full-uid.json -- \
  --config /absolute/research-config.json --binary /absolute/native-prefill \
  --helper-source research/glm53_batched_prefill/native_prefill.cpp \
  --prompts /absolute/new-prompts/native-prompts.json \
  --runtime-libraries /absolute/runtime/bin --output /absolute/new-full \
  --microbatch 8 --mode full --diagnostic --max-seconds 5400
```

Le [lanceur UID](UID_SUPERVISOR.md) acquiert le verrou du modèle avant le
wrapper et transmet son descripteur au helper. Ses tests utilisent des processus
synthétiques ; aucune exécution matérielle de ce nouveau lanceur n'est encore
revendiquée. Il conserve les sources et la clé du wrapper gelé.

Pour 8/16, comparer directement `full` puis `restore` : le comparateur exige
les trois états à 208 identiques et la correspondance SHA256/taille du snapshot,
ainsi que les suffixes et continuations. L’alignement seul ne suffit pas.
Pour les lots 32 et supérieurs, le protocole de découpage à 208 est encore
structurellement incomplet : lire [l’audit des frontières](split32_protocol_audit.md)
avant toute expérience. Le lanceur accepte `split`, mais cela ne qualifie ni les
frontières ni les checkpoints du helper actuel. Les reçus
`guard.json` doivent être complets et valides. Aucun second moteur physique
ne doit être lancé en parallèle. Le diagnostic écrit les
états et peut consommer de l'espace disque ; il ne fournit pas une mesure
non instrumentée de performance. Le comparateur `batch_compare.py` aligne les valeurs par position de token
et exige les checkpoints 128/256/384/512/final communs, sans confondre des
écritures de tailles différentes.

Pour la qualification fonctionnelle indépendante du cache :

```sh
$PYTHON -m research.glm53_batched_prefill.campaign \
  --config /absolute/research-config.json --output /absolute/new-agents \
  --mode agents --microbatch 8 --tokens 512 --reasoning-effort native \
  --agent-order 1,600,1 --request-seconds 2700 --max-seconds 7200
```

Le contexte doit couvrir chaque prompt complet et son budget de sortie.
La campagne courante utilise 2 048 tokens, y compris tous les témoins
comparables. Le slot ne tronque pas les historiques pour faire passer un
essai. RSS 8,5 Gio, croissance de swap 32 Mio, pression, réserves système,
secteur, couvercle, horloges, identité PID et annulation sont conservés.

Un passage en production exige les preuves matérielles indépendantes A/B/C,
la revue des réponses utiles et des sorties jusqu'à EOS. Les fixtures et
la CI ne qualifient ni ces décisions ni l'efficacité de 600 agents.

## Premier essai fonctionnel clôturé

Le [témoin CPU8 natif Max](evidence/agents-cpu8-timeout/summary.json) traite
522 tokens réels en 1 238,016 s puis conserve 171 IDs avant `REQUEST_TIMEOUT`
à 45 min, sans fermeture du raisonnement ni EOS. L’agent 600 n’est pas lancé.
La décision reste `failed`, avec sortie partielle ; le retour natif −9 est
observé pendant le nettoyage après délai. Le pic RSS échantillonné est
5,616 Gio et la croissance de swap observée est nulle. Les alertes de pression 2
restent dans le domaine accepté du watchdog existant avec réserve/marge/swap.
Les sources mesurées et preuves brutes sélectionnées sont archivées séparément
du pilote durci publié. Aucun gain n’est calculé entre ce prompt et les UUID
différents du diagnostic natif.

Le comparateur de reçus réutilise le watchdog existant. Il accepte uniquement
une réserve permanente, et en pression 2 la réserve augmentée de la marge
d’alerte ainsi qu’un swap sous le seuil d’alerte ; les grâces ne servent pas
à contourner ces bornes conservatrices. RSS 8,5 Gio, swap 32 Mio et les critères
numériques restent identiques.

## Témoin natif CPU8 clos et rejeu exact

Le [témoin complet](evidence/native-full8-closed/README.md) termine les trois
préfills gelés `1 → 600 → 1` en 1 285,763678 / 1 322,539045 / 1 314,183277 s,
avec quatre vrais tokens greedy diagnostiques par requête. Le code de sortie
est 0 ; secteur, intégrité et gardes sont validés à la clôture. Le pic RSS
échantillonné est de 5 608 177 664 octets et la croissance de swap observée est nulle.

Le rejeu de l'agent 1 compare exactement ses 154 880 logits finis, son état
sérialisé, ses captures et ses 15 642 écritures d'état observées ; la continuation
et son post-état sont aussi identiques. Cette vérification interne au même bras
ne remplace pas une comparaison indépendante avec un cache ou un autre batch.
Les trois états complets à 208 sont identiques et mesurent 156 739 588 octets.
Aucune restauration n'a encore été exécutée.

Le verrou UID de ce témoin a été acquis par un guardian externe après le
lancement, sans héritage rétroactif de descripteur. Son reçu conserve cette
limite. Le nouveau lanceur apporte une protection avant lancement pour les
prochains essais, sans modifier cette chronologie historique.

## Composants Metal représentatifs clos

Après le refus initial sur batterie, conservé dans les preuves, le retour
sur secteur permet deux [mesures de composants](evidence/metal-representative-closed/README.md)
closes, sans chargement de GLM. La projection Q8_0 synthétique **1536 × 4096**
produit **380 928 sorties finies exactes** face à ARM `nrc=2` et au graphe
CPU10, pour N=8/16/32/64/128. Le [composant Q2_K](metal_q2_audit.md), un expert
synthétique **2048 × 4096**, produit **114 688 sorties finies exactes** face à
ARM `nrc=1`, pour N=8/16/32. Les deux campagnes conservent les sources et le
swap système observé ; leurs reçus et journaux sont sélectionnés avec SHA/tailles.

Les temps GPU portent sur des données préparées avec trois répétitions.
L'oracle Q2 n'est pas un témoin de vitesse CPU10 et sa quantification CPU est
hors des horodatages GPU. Ces résultats ne démontrent aucun gain de préfill
GLM, aucune lecture SSD amortie ni fidélité du graphe complet : l'objectif A
reste non qualifié, ainsi que le snapshot réutilisé et les deux réponses
utiles jusqu'à EOS.

La [reconstruction V4](evidence/source-reconstruction/README.md) vérifie les
empreintes de **13 sources natives** après application ordonnée de 16 patches
dans un nouveau checkout. Elle ne reconstruit aucun binaire et ne transfère
aucune qualification aux prochaines bibliothèques.
