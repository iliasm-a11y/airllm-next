# Préfill partagé et deux agents GLM sur M5 : résultat borné

3 novembre 2026. **La réutilisation exacte du préfill reste désactivée et les
deux agents ne sont pas qualifiés jusqu'à EOS.** La campagne réelle a été
arrêtée par les protections mémoire, avant toute requête d'agent. Aucun gain
de latence, de lectures ou de débit n'est revendiqué.

Le travail apporte une mesure native du vrai préfixe, l'audit des états, un
témoin de recherche explicitement sans cache et un pilote de qualification
conservant les échecs. Il part de PR #20 `a980f69c40e9170aef411fb4333613d2f7c6415a`,
incluant PR #21 `8138298f927dd17f583b5c0caed884ea432c4c01`. Les deux têtes ont
été vérifiées sur GitHub avant le travail. La nouvelle branche est
`research/glm53-shared-prefill-qualification`, empilée sur #20 ; aucune fusion
ni modification de `main` n'est effectuée.

## Préfixe établi token par token

Le vrai constructeur `AgentContexts.begin` a généré les messages complets des
600 identités dans une base temporaire. Les tâches utiles des agents 1 et 600
sont celles du pilote : consommation de stock et comparaison de devis, avec
deux mémoires et marqueurs distincts. Les autres identités conservent leurs
rôles et instructions RONIN. Aucun système, historique ou message n'a été
tronqué. Il s'agit de premiers tours complets, pas d'une preuve sur 600
historiques arbitraires.

Le helper natif charge seulement le vocabulaire du petit premier GGUF ; il
ne crée aucun contexte d'inférence. Il applique le Jinja natif puis tokenise
chaque **prompt complet** avec le runtime V4 épinglé. Les octets rendus et
tous les IDs sont exactement égaux à ceux du compagnon pour les 600 prompts,
puis pour les deux prompts historiques relus dans la base isolée du précédent
probe. Ce dernier contrôle retrouve 447 et 448 tokens complets.

| Périmètre | Résultat établi |
|---|---:|
| Préfixe du texte système | 832 caractères / 869 octets UTF-8 |
| Préfixe du prompt rendu, en-tête compris | 885 caractères / 922 octets UTF-8 |
| Plus long préfixe d'IDs natifs des 600 prompts complets | **211 tokens** |
| Même comparaison pour les deux prompts historiques | **211 tokens** |
| Entrées complètes des nouvelles tâches agents 1 / 600 | **523 / 544 tokens** |
| Frontière arithmétique à une microbatch de 8 dans ce préfixe | **208 tokens**, candidat non qualifié numériquement |

Les 211 tokens comprennent l'en-tête de chat. Ils ne sont pas une conversion
directe des 869 octets du système. Les trois tokens communs suivants une
frontière à 208 resteraient à calculer avec chaque suffixe. Le bénéfice en
temps ne peut pas être déduit d'une fraction de tokens.

Le gabarit natif et le fichier compagnon ont des SHA différents : le fichier
compagnon contient un saut de ligne final supplémentaire. Leurs rendus sont
égaux sur tous les cas mesurés. Les deux empreintes restent distinctes dans
les preuves ; cette égalité n'est pas généralisée à d'autres options ou outils.

La dernière exécution du helper sur les 600 prompts dure 5,217296 s, avec un
pic RSS échantillonné de 167 706 624 octets et un maximum observé de
17 829 888 octets de lectures comptées par macOS pour ce PID. Ces chiffres
couvrent **rendu/tokenisation et écritures du JSON**, dans cet état de cache
système ; ce ne sont ni un préfill GLM, ni des lectures NAND, ni un comparatif
d'inférence. Le premier GGUF de 9 430 112 octets est haché ; aucun nouveau
scan SHA du deuxième shard de 110 Go n'est revendiqué.

## Réutilisation exacte : décision et contrat

L'[audit des états](STATE_AUDIT.md) conserve dix sources natives dont les
empreintes ont été rapprochées de la reconstruction des 16 patches du
manifeste V4. Il identifie attention MLA, cellules/positions, KDA R+S,
indexer/DSA et layout couplé, métadonnées hôtes et sampler. Les activations
mHC et routes sont locales au graphe dans cette implémentation ; elles
restent des cibles de comparaison après reprise, sans inventer un cache
temporel supplémentaire.

Le point bloquant principal est que les checkpoints ordinaires
`PARTIAL_ONLY` peuvent **omettre l'attention et l'indexer**, supposés encore
présents dans le contexte actif. Ils ne sont pas un snapshot autonome après
reset complet. Le cache RAM générique consomme aussi son entrée au restore.
Activer `cache_prompt` ou retirer le suffixe KV d'un agent traité ne satisfait
donc pas le contrat demandé.

Une voie autonome devrait capturer avant toute identité privée, sérialiser
toute la séquence avec les flags complets appropriés, conserver les positions
et la clé de provenance, puis restaurer sous le verrou du slot. Elle devrait
comparer recalcul complet, découpage sans cache et restauration, avec les
mêmes frontières de microbatch, logits complets finis, routes ordonnées,
KDA, attention/indexer et activations utiles. Erreur de restore, annulation,
changement de poids/template/runtime ou état incohérent doivent invalider la
base et vider les états cohérents.

**Aucune telle restauration n'est implémentée ou activée ici**, car son
exactitude, sa taille et son coût net n'ont pas pu être mesurés. Cela ne
démontre pas une impossibilité technique. Le [protocole comparatif](COMPARISON_PLAN.md)
fixe un candidat borné : un seul snapshot immuable, plafond de recherche
256 Mio avant allocation, un état actif, aucune copie par identité et un
budget global incluant scratch/copies/caches. Les seuls R+S logiques estimés
font déjà 145,5625 Mio ; cette estimation n'est pas une taille sérialisée
mesurée. Le profil effectivement exécuté a **zéro cache de préfixe demandé**.

## Campagne matérielle et audit des échecs

MacBook Air M5 CPU10, 16 Gio, SSD interne. Le Mac était initialement sur
batterie ; les opérations d'inférence n'ont été lancées qu'après branchement
au secteur confirmé. Le contexte de recherche est explicitement 1 024,
batch/microbatch 8, sortie maximale 256 tokens par tâche. 523+256 et 544+256
tiennent dans cette limite. Le passage à 1 024 conserve les prompts complets ;
il n'est pas un raccourcissement destiné à faire passer l'essai.

Le pilote conserve RSS 8,5 Gio, hausse globale du swap 32 Mio, pression,
réserves, intervalles et délais du watchdog existants ou plus stricts,
intégrité des shards et bibliothèques, annulation et arrêt possédés. Il
utilise un registre neuf de 600 identités, CPU10/GPU0, un seul modèle prévu et
un slot. Le profil de recherche ajoute explicitement :

```text
--cache-ram 0 --ctx-checkpoints 0 --no-cache-idle-slots
--cache-reuse 0 --no-cache-prompt --spec-type none
```

Les requêtes prévues gardent aussi `cache_prompt=false` et `id_slot=0`.
Ce profil diffère des defaults natifs historiques de #20 ; les anciens temps
ne sont pas présentés comme son témoin contemporain. Le mapping nocache et
les préchargements du profil expérimental CPU V4 sont conservés explicitement.

| Essai | Durée totale hôte | Résultat et portée |
|---|---:|---|
| CPU sans cache, tentative 1 | 8,006217 s | `SWAP_LIMIT` pendant chargement ; +39 845 888 octets = **38 Mio**, au-dessus de 32 Mio ; RSS max échantillonné 3 829 415 936 octets ; PID 6046 sorti à **−15** sous garde |
| Observation après nettoyage | 30 s, 7 relevés | Secteur/couvercle acceptés ; swap constant à 1 341 849 600 octets ; pression **2 à chaque relevé** ; mémoire disponible estimée entre 4,900 et 4,991 Go |
| CPU sans cache, tentative 2 | 1,903052 s | `MEMORY_ADMISSION` avant démarrage natif ; aucune génération, aucun serveur supplémentaire |

Le secteur est stable sur les observations des deux essais. Les poids et les
artefacts natifs sont vérifiés via reçus et identités inchangées. La première
tentative n'atteint pas l'état prêt : **zéro chargement natif achevé, zéro
requête, zéro ID de sortie et zéro EOS**. La seconde est refusée avant tout
démarrage. Aucun processus possédé ne subsiste au contrôle de clôture.
La hausse du swap est globale et n'est pas attribuée intégralement au serveur.

La version mesurée enregistrait l'argv seulement après chargement réussi et
créait l'observateur I/O après celui-ci. Le premier échec ne possède donc pas
d'argv natif brut observé ni de comparatif I/O d'inférence : les flags ci-dessus
sont établis par le code gelé, la configuration et l'aide acceptée, pas par
une observation d'argv de ce processus. La version livrée améliore ensuite
la journalisation des échecs ; les rapports bruts ne sont pas réécrits et la
version améliorée n'est pas présentée comme rejouée sur ce modèle.

Le [protocole du pilote](QUALIFICATION.md) exige pour une réussite fonctionnelle
deux EOS naturels, deux réponses finales après réflexion, des calculs utiles,
des marqueurs isolés dans les prompts natifs et sorties, un transcript d'IDs
complet et toutes les gardes acceptées. Il distingue cette réussite d'une
preuve numérique de reset/restauration. Ici, les deux restent **non qualifiées**.
L'[audit détaillé](FAILURE_AUDIT.md) décrit les échecs et limites d'observation.

## Comparatif, tests et livraison

Le [comparatif structuré](evidence/comparison.json) laisse les champs de
préfill, décodage, sérialisation, restauration, lectures d'inférence et gain
net à `null` : ils ne sont pas mesurés. Les 937,169 s historiques concernaient
un autre prompt/profil et ne permettent aucun gain contemporain. Aucun
logit, routage ou état numérique nouveau n'est inventé pour un chargement
interrompu avant inférence.

Les journaux et le [récapitulatif des tests](evidence/validation-summary.json)
séparent les nouveaux contrôles portables des suites RONIN et backend native.
Le helper C++ est compilé et exécuté réellement avec les bibliothèques V4 pour
le contrôle complet des prompts. La CI ajoute la nouvelle suite portable ;
elle ne constitue pas une qualification matérielle.

Les preuves compressées dans `evidence/` sont sans perte et leur sélection
conserve les hashes des originaux et archives. La livraison locale contient
aussi les JSON complets, journaux natifs, bases isolées, reçus, sources gelées,
tests, patch et archive de code. Aucun poids n'est dupliqué. **CPU10/GPU0 reste
la voie de production existante ; aucun cache ou candidat GPU nouveau n'y est
activé.** Une nouvelle campagne native d'états et deux EOS restent nécessaires
avant d'aller plus loin.
