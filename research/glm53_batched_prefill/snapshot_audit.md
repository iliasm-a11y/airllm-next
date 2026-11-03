# Snapshot complet du préfixe : code expérimental et limites

Le nouveau `snapshot.h` appelle réellement les API publiques de séquence avec
`LLAMA_STATE_SEQ_FLAGS_NONE`. Il ne modifie ni le runtime natif, ni les poids,
ni le backend RONIN installé. Ses tests de contrat ne constituent pas une
qualification GLM : ils emploient des API natives factices compilées. Seule
une comparaison matérielle complète et admissible peut qualifier la reprise.

## Ce qui doit être sérialisé

Les sources inspectées sont le checkout natif V3 fondé sur llama.cpp
`3f8af15a76ac8675812e856da45454bafc90f12c`, avec les patches de recherche
préexistants. PR22 avait déjà vérifié l'identité des fichiers d'état
sélectionnés avec sa reconstruction V4. Toute nouvelle compilation ou
modification impose sa propre vérification ; cette lecture ne transfère
aucune qualification à un nouveau binaire.

| Source native inspectée | Constat |
|---|---|
| `src/llama-context.cpp:3531` | La requête de taille compte la mémoire de séquence et un petit en-tête. Le helper synchronise explicitement le contexte avant cette requête. |
| `src/llama-context.cpp:3783` | Le snapshot de séquence appelle seulement `memory->state_write`. Il ne sauvegarde ni logits de sortie, ni sampler, ni historique de chat. |
| `src/llama-context.cpp:4696` | Les copies d'état publiques synchronisent le contexte avant écriture/lecture. Le nombre d'octets retourné doit être vérifié. |
| `src/llama-memory-hybrid-idx.cpp:253` | Écriture : attention, récurrent, puis indexer. La lecture adopte le layout de cellules de l'attention pour l'indexer et invalide son cache KPool. |
| `src/llama-memory-hybrid-idx.cpp:295` | Un échec partiel vide les états couplés ; le nouveau helper ajoute un reset complet avec `clear(true)` après tout refus de restauration. |
| `src/llama-memory-recurrent.cpp:783` | Les données R+S sauvegardées correspondent aux lignes logiques actives, y compris la bonne plane de rollback, plutôt qu'à une supposition sur la ligne physique courante. |
| `src/llama-memory-recurrent.cpp:864` | Le lecteur valide dimensions/types, restaure les cellules, puis remet l'index de rollback à zéro. |
| `src/llama-kv-cache.cpp:2070` | `PARTIAL_ONLY` saute l'attention sans SWA ; l'indexer suit cette même politique. Ce flag demeure interdit pour le snapshot autonome. |
| `src/llama-memory-hybrid.cpp:173` | Le minimum hybride est le maximum des minima attention/récurrent. Exiger artificiellement `pos_min==0` rejetterait un état KDA normal. Le helper contrôle `pos_max==207` à la frontière de208. |
| `src/models/glm5-next.cpp:699` | mHC et les routes/activations du graphe ne sont pas un cache temporel indépendant. Ils restent des témoins numériques à comparer. |

La seule taille R+S précédemment dérivée, **152633344 octets**, ne donne
toujours pas la taille mesurée du snapshot complet.

## Borne et admission

Le helper garde un seul tableau propriétaire, non copiable et non déplaçable.
Une deuxième capture est refusée tant qu'il existe. La taille native est
interrogée avant allocation et doit rester dans **256Mio**. Une clé SHA256
canonique doit lier les shards originaux, tokenizer, template, runtime et
bibliothèques, IDs/positions du préfixe, paramètres effectifs et options de
rendu. `snapshot_key.py` valide le schéma et couvre tous ces champs ; il ne
vérifie pas à lui seul la provenance des valeurs qu'un lanceur lui fournit.

La copie hôte n'impose pas un second tampon de la taille du snapshot :
`llama_io_write_host` (`src/llama-context.cpp:2987`) réserve les destinations
dans le tableau fourni, puis `ggml_backend_tensor_get_n` y transfère les
tenseurs. `llama_io_read_host` (`:3059`) fait le transfert inverse. Les
vecteurs temporaires conservent des descripteurs, pointeurs, tailles et
offsets. `ggml/src/ggml-backend.cpp:607-751` peut aussi grouper les descripteurs
par type de buffer. Cela diffère du chemin de sérialisation **vers fichier**,
qui emploie un tampon par tenseur. Cette distinction ne prouve pas une
empreinte temporaire nulle ni une valeur de scratch mesurée.

`native_scratch_bound` est une réserve séparée exigée par le helper, soumise
avec le payload au garde fourni par le lanceur. Sa provenance doit être
documentée. Une réserve prudentielle n'est pas une mesure de scratch. Le
garde global conserve l'état actif, buffers de calcul/capture, snapshot et
scratch dans son budget et surveille RSS, footprint, réserve système,
pression et swap. Une capture diagnostique d'état final crée temporairement
un deuxième tableau : elle doit être admise séparément et libérée avant la
requête suivante. Elle n'est pas une copie conservée pour chaque identité.

## Contrat de continuation et d'échec

La sauvegarde intervient avant toute identité/mémoire privée et avant
initialisation du sampler. Avant chaque restauration, le contexte est
entièrement vidé avec `clear(true)`. La copie est contrôlée par son SHA256,
la clé, la taille lue et la position finale attendue. Le snapshot reste
immuable après les trois restaurations.

Un suffixe non vide est obligatoire : il doit calculer les nouveaux logits
avant le premier échantillonnage. Le snapshot ne remplace pas un checkpoint
de génération. Le sampler, le parser et l'historique d'acceptation demeurent
propres à chaque requête. Annulation, changement de clé, taille courte,
corruption ou refus du garde invalident le snapshot et effacent le contexte.
Il n'y a ni reprise automatique, ni restauration partielle tolérée.
Les refus de capture après admission de l'identité du contexte, y compris
taille excessive et refus avant allocation, effacent également l'état actif.
Un contexte refusé par le contrôle initial n'est pas modifié. Une tentative
de remplacer une base immuable existante est refusée en conservant cette base.

## Preuve numérique exigée

`native_compare.py` distingue le témoin complet du témoin découpé, puis le
témoin découpé de la restauration. Les frontières et IDs des prompts doivent
coïncider. Pour les restaurations, les comparaisons concernent les suffixes
et les trois états/logits finaux ; le préfixe commun n'est pas recompté comme
trois calculs indépendants. Le comparateur exige notamment :

Une comparaison directe complet/restauré évite un recalcul découpé redondant
uniquement pour les microbatches8/16, où208 est déjà une frontière native.
Elle exige en plus les trois checkpoints complets de référence à208,
identiques entre les requêtes, puis l'égalité du snapshot immuable avec leur
taille et SHA256. L'alignement sans cette preuve ne suffit pas. Le comparateur
de tailles de batch conserve ses checkpoints128/256/384/512/final ;208 est
comparé en supplément seulement lorsqu'il est présent dans les deux bras.

- Trois préfills complets dans l'ordre1,600,1, un modèle et un slot.
- Des reçus de gardes claires, limites RSS8,5Gio/swap32Mio, sorties natives0,
  intégrité et continuité secteur/couvercle explicitement établies.
- Tous les logits finis, les fichiers d'états complets et leurs empreintes.
- Par microbatch comparable, les routes ordonnées et routeurs des42 couches
  MoE3..44 et les sorties KDA des34 couches correspondantes.
- Comparaison des éléments logiques selon leurs strides, avec les critères
  CPU bit à bit préexistants, sans comparer arbitrairement le padding.
- Parité ordonnée des écritures `CPY`/`SET_ROWS` observées : descripteurs,
  formes, longueurs logiques et SHA256 finis, avec couverture des90 cibles
  KDA R/S, attention et indexer à chaque microbatch comparée. Cette vérification
  ne revendique pas les écritures ou temporaires non capturés.

L'absence d'un même tenseur sur les deux côtés ne produit pas un faux succès.
Un contrôle interne à chaque bras compare aussi les requêtes0 et2 : mêmes
IDs de prompt, logits complets, états sérialisés, captures numériques et
écritures observées, en normalisant les numéros de requête. Pour le bras
restore, seules les captures du suffixe effectivement recalculé sont
comparées. Ce rejeu du même input natif évite de qualifier deux bras qui
reproduiraient le même défaut de reset. Il ne qualifie pas l'isolation des
historiques ou mémoires privées d'une session RONIN complète.
Une continuation diagnostique doit aussi échantillonner jusqu'à4 vrais tokens
avec un sampler natif greedy neuf, ou s'arrêter sur un EOG natif antérieur.
Chaque étape fournit ses logits complets finis ; le token doit correspondre
à leur argmax et la première étape doit retrouver les logits de fin du
préfill. Les décodes des étapes1..3 fournissent routes/routeurs/KDA,
`result_output`, écritures observées et un état final brut. Les comparateurs
exigent leur parité entre bras ainsi qu'entre les requêtes0 et2 de chaque
bras. Un arrêt au budget4 reste `diagnostic_budget`, jamais une mission
terminée. Même un EOG natif sur cette courte continuation ne qualifie pas
une réponse finale utile ni l'objectif fonctionnel des deux agents.
Des reçus incomplets, un trou de microbatch, NaN, payload tronqué, absence de
source brute ou arrêt par garde produisent un refus. Cette qualification
finie n'inclut pas les états/opérations non capturés, une génération jusqu'à
EOS, l'isolation des historiques RONIN ni un gain de performance.

## Empreintes de la lecture native

```text
33d8cf6d489cce950bf35bc19979f90f87b2e987e9fc095875ba8cf7debedcb4  src/llama-context.cpp
0c69f0f91072192d71f23ff1d920379aa8b90799db4891c55175db81407d0c37  src/llama-memory-hybrid.cpp
08cbd47bcac5e481ef66c586c27510b210f34d76042dbafa6d51d9dbc489831a  src/llama-memory-hybrid-idx.cpp
2c5e97e0cdc1b50950735f164666bbd974d4ae01e9dbab224986f3f5901948ae  src/llama-memory-recurrent.cpp
40205d197380761b373658d6206f4ccd91455b4f2f45a3e35c18c3f9ea0ee6f3  src/llama-kv-cache.cpp
ab8ddd13fa5c28de863392580f443584fefdeb020976675445496ce655f0f60b  src/models/glm5-next.cpp
dce7b42ecdba028d7c7ee2b979eedcf2277a129b4f0d8738fcf42b929c3a00b9  ggml/src/ggml-backend.cpp
f07ffbabd3fbe9ef21d97b4d4ce8328d9c2a123ccbf5266f04a3146d70ce4f91  include/llama.h
```

Les tests portables emploient des états factices, y compris la fixture C++
qui remplace les API natives. Ils couvrent notamment ordre, empreinte,
statut, valeurs finies, continuation incomplète et un même défaut de reset
reproduit par les deux bras. Aucun chargement du vrai modèle ni temps GLM
n'est compté dans cette validation portable ; le journal de tests conserve
le nombre final après les extensions du comparateur.
