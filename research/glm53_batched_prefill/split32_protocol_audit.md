# Audit statique borné du protocole split/restore à 208 pour microbatch 32+

**Conclusion : le chemin actuel ne permet pas de qualifier `full32 → split32 → restore32` contre le préfill complet.** Le premier lien est rejeté structurellement par le comparateur, et le découpage produit des checkpoints incompatibles avec le comparateur de tailles. Cela ne démontre aucune divergence numérique du modèle : aucune inférence, compilation ou exécution GPU n'a été faite pour cet audit.

**La prochaine action CPU16 puis restore16 reste inchangée**, sous réserve des admissions secteur/ressources et de la parité réellement mesurée. 208 est multiple de 16, mais pas de 32/64/128. Ne pas modifier ou reconstruire les sources gelées pour cette découverte.

## Sources lues

Audit du checkout local `work/publication`, limité aux sources et fixtures présentes au moment de la lecture. Les lignes sont relatives à `research/glm53_batched_prefill/` ; elles ne prétendent pas attester un nouveau commit distant.

| Source | Contrat constaté |
|---|---|
| `native_prefill.cpp:186-198` | Chaque appel de `decode(first,end)` repart de `first`, avec `take=min(ub,end-offset)`. Le suffixe repart donc exactement de 208. |
| `native_prefill.cpp:199-208` | Checkpoint seulement après un decode fini, si la position est multiple de 128, égale 208, ou égale la longueur totale du prompt. |
| `native_prefill.cpp:212-228` | Préfixe/snapshot et reset/restore sont codés à 208. `split` calcule séparément 0..208 et 208..fin ; `restore` ne recalcule que le suffixe après restauration. |
| `native_compare.py:332-354` | `full → restore` direct exige `208 % microbatch == 0`, trois états complets de référence à 208 identiques et le SHA 256/taille correspondant du snapshot. |
| `native_compare.py:497-512` | `full/split` est un couple de modes admis, mais exige ensuite les mêmes triplets `(request,offset,tokens)` sur tous les lots ; `split/restore` ne compare ces triplets qu'à partir de 208. |
| `batch_compare.py:47-77` | Checkpoints 128/256/384/512 jusqu'à la longueur du prompt, plus état final, obligatoires. 208 n'est facultativement admis que si aligné sur la taille de microbatch. |
| `batch_compare.py:170-184` | Les modes doivent être `full/full`. Autoriser simplement `full/split` ne réparerait donc pas les checkpoints manquants. |
| `test_native_compare.py:87-98,129-142` | La fixture positive full/split et split/restore utilise microbatch 16 et un prompt 216 tokens. Elle ne génère pas le découpage réel d'un split32. |
| `test_native_compare.py:429-434` | Le cas `unaligned` vérifie le rejet du full/restore direct avec un profil déclaré 32 ; il ne qualifie aucun split32. |
| `test_batch_compare.py:91-99,179-191` | Le cas complet positif compare 8/16 ; le test 208 ne remplace pas les checkpoints obligatoires. |

Empreintes SHA 256 des trois sources auditées :

```text
558aa441ed74aca6b295613ccb414c4b1f2bf2134a6bdc0ae32a09b35c32c395  native_prefill.cpp
0019976ae27a7296e2bdafea2d67ff4a77dd77c77a6586b838d7da0ea629d6de  native_compare.py
22c8a252dbb8cedbe2de1b82d3c11768e2a2bf1b0ad6abe8eeea3ebc57644f84  batch_compare.py
```

Le superviseur UID actuel accepte désormais `full`, `split` et `restore` (`uid_supervisor.py:42,138`). Il n'est donc pas un obstacle opérationnel actuel au lancement d'un split. Cette admission ne répare pas le protocole de fidélité décrit ici et ne modifie pas le témoin historique.

## Exemple exact des frontières, sans calcul du modèle

Les listes suivantes résultent uniquement de l'arithmétique de la boucle C++, pour le prompt gelé de 521 tokens. Les intervalles sont demi-ouverts : `[début,fin)`.

### Microbatch 32

`full32` :

```text
[0,32) [32,64) [64,96) [96,128) [128,160) [160,192)
[192,224) [224,256) [256,288) [288,320) [320,352)
[352,384) [384,416) [416,448) [448,480) [480,512) [512,521)
```

`split32`, puis suffixe de `restore32` :

```text
préfixe : [0,32) [32,64) [64,96) [96,128) [128,160) [160,192) [192,208)
suffixe : [208,240) [240,272) [272,304) [304,336) [336,368)
          [368,400) [400,432) [432,464) [464,496) [496,521)
```

Premier désaccord structurel : à l'offset 192, `full32` envoie 32 tokens et `split32` en envoie 16. La suite est décalée de 16 tokens. Le comparateur de restauration rejette donc les reçus, **même dans l'hypothèse où toutes les valeurs numériques seraient égales**. D'autres contrôles pourraient naturellement échouer avant cette porte ; elle suffit à empêcher une réussite.

Après 208, les fins de suffixe sont `208 + k*32`, puis la fin privée. Comme 208 n'est pas multiple de 32, aucune de ces positions n'est multiple de 128. Les checkpoints 256/384/512 ne sont donc jamais atteints. Les checkpoints émis pour cette requête sont :

| Mode | Checkpoints réels issus de la règle actuelle |
|---|---|
| full32 | 128,256,384,512,521 |
| split32 | 128,208,521 |
| restore32 | Préfixe initial :128,208 ; suffixe privé :521 |

Pour le prompt 542, remplacer seulement la fin privée 521 par 542 ; le défaut de frontières communes demeure.

### Autres tailles

| Microbatch | Frontières autour du partage 208 | Effet |
|---:|---|---|
| 8 | 200→208→216 | Même grille que full ; le protocole direct reste structurellement possible. |
| 16 | 192→208→224 | Même grille que full ; les points 256/384/512 sont atteints. |
| 32 | full192→224 ; split192→208→240 | Changement de forme et décalage du suffixe. |
| 64 | full192→256 ; split192→208→272 | Même problème ; après 208, les fins sont 272/336/400/464/fin. |
| 128 | full128→256 ; split128→208→336 | Même problème ; suffixe 336/464/fin. |

Dans ces exemples, `split/restore`32+ peuvent avoir **le même découpage entre eux**. Une éventuelle parité de cette paire prouverait seulement la restauration relativement au calcul déjà découpé. Elle ne rétablit pas la preuve manquante contre `full32` et ne permet pas de promouvoir le cache au contrat de préfill complet.

## Ce que les outils actuels peuvent ou ne peuvent pas conclure

1. `batch_compare(full8,full32)` reste un protocole prévu : les deux modes sont full, les opérandes sont recollés par token, les checkpoints communs sont atteints. Aucun succès matériel n'est déduit de cette possibilité.
2. `native_compare(full32,split32)` ne peut pas passer avec le helper courant : les reçus de lots diffèrent.
3. `native_compare(full32,restore32)` direct est explicitement refusé par l'alignement 208.
4. `batch_compare(full32,split32)` est refusé sur le mode ; ignorer ce premier refus laisserait des checkpoints obligatoires manquants et un 208 non admis pour 32.
5. `native_compare(split32,restore32)` n'est pas une preuve complète du cache contre le témoin full32, même s'il terminait et passait sur son périmètre.

La formulation documentaire « full/split précède split/restore pour les autres tailles » exprime donc un protocole cible encore incomplet dans l'outillage courant. Il ne faut pas lancer une longue campagne 32+ en présumant que cette chaîne est actuellement qualifiable.

## Solution future conservatrice : une frontière de partage alignée

**Recommandation si l'on conserve les contrats actuels d'identité des lots et d'écritures ordonnées :** rendre la frontière de préfixe explicite et compatible avec la grille du microbatch, dans une nouvelle branche et de nouveaux binaires.

Avec un LCP réellement confirmé de 211 IDs :

| Microbatch | Frontière alignée candidate dans le LCP |
|---:|---:|
| 8 /16 | 208 |
| 32 /64 | 192 |
| 128 | 128 |

À 192, full32 a déjà fini exactement le lot `[160,192)` ; le suffixe commence par `[192,224)`, comme le témoin. À 192 pour 64, le suffixe suivant est `[192,256)`, également identique au témoin. Le même raisonnement vaut à 128 pour 128. Cela élimine le désaccord structurel sans modifier une réduction, une tolérance ou un critère de parité.

Cette proposition demande néanmoins une **implémentation et une qualification nouvelles**, pas une activation :

- Remplacer les constantes 208 du helper et des comparateurs par une frontière déclarée, validée contre les vrais IDs communs ; conserver leurs positions absolues.
- Faire émettre au témoin full les checkpoints à cette frontière explicite **en plus** de 128/256/384/512/final, sans supprimer de couverture existante.
- Lier la frontière et sa politique aux tokens/positions du manifeste et à la clé. Ne pas faire ignorer un changement de préfixe par le comparateur de dépendances ; les deux bras d'une comparaison doivent partager la même définition du préfixe.
- Garder la sérialisation complète, la base unique, le plafond 256 Mio, le reset, le sampler neuf, les critères CPU bit à bit, les routes ordonnées, les écritures observées, les post-états et le rejeu 0/2.
- Reconstruire avec une nouvelle provenance, puis refaire les témoins et tests adaptés. L'alignement est une condition de forme, jamais une preuve d'égalité après restauration.

Un partage 192 ne doit pas être appelé « cache 208 ». Il réutilise 16 tokens de moins par suffixe ; son coût net réel reste à mesurer. Une frontière commune 128 pour toutes les tailles serait aussi cohérente, mais économiserait moins de travail. Aucun gain de l'une de ces variantes n'est estimé ici.

## Si 208 reste une exigence pour 32+

Ce serait un protocole distinct, plus complexe. Une possibilité de découpage futur serait de revenir à la grille absolue après 208 : pour 32, suffixe `[208,224)` puis `[224,256)`, etc. Les checkpoints 256/384/512 seraient de nouveau accessibles. Cependant le lot complet `[192,224)` resterait remplacé par deux demi-lots ; **cela ne répare pas le comparateur actuel** et ne garantit aucune fidélité KDA.

Il faudrait alors une comparaison full/split explicitement conçue pour des partitions différentes : alignement logique par token, toutes les valeurs finies aux critères inchangés, états persistants complets aux mêmes frontières communes et continuations exactes. Les événements physiques d'écriture ne peuvent pas être déclarés identiques par simple égalité de compte ou en supprimant ceux qui diffèrent. Leur portée devrait être établie séparément, comme pour le protocole de tailles, sans présenter une parité finale comme une preuve universelle des intermédiaires.

L'API actuelle ne fournit pas magiquement un état autonome à 208 après un lot full32 qui a déjà avancé jusqu'à 224. Aucun état manquant ne doit être inventé. Si l'on exige le contrat d'identité des lots et de chaque écriture intermédiaire du protocole actuel, retenir la frontière alignée 192 est la voie conservatrice ; conserver 208 demanderait d'abord une nouvelle définition de preuve suffisamment complète.

## Conséquence immédiate

**Aucun changement de la prochaine action.** Garder les sources CPU16/restore16 gelées et attendre une admission secteur/ressources valide. Qualifier full16 contre son témoin pertinent, puis restore16 contre full16 si les données le permettent. Ne pas augmenter les lots, supprimer des checkpoints ou relâcher une tolérance pour contourner cette incompatibilité.

Les futurs essais cache 32/64/128 restent bloqués par un défaut de protocole indépendant des performances matérielles, jusqu'à une nouvelle conception explicite. Cet audit n'a modifié aucun helper, wrapper, comparateur ou test, et n'a exécuté aucun modèle, GPU ou build.
