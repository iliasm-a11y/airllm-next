# Composants Metal représentatifs mesurés et clos

Les deux campagnes de composants se terminent le 3 novembre 2026, sous
secteur et pression 1, avec sources identiques avant/après. Les observations
sélectionnées du swap système sont identiques au début et à la fin.
Aucun modèle GLM ni poids de son checkpoint n'est chargé ou lu : poids et
activations sont synthétiques, aux dimensions de matrices observées du modèle.

## Q8_0 : projection complète 1536 × 4096

Les **380 928 sorties finies** des cinq tailles sont identiques bit à bit à
la routine ARM `nrc=2` et au graphe GGML CPU10. Il s'agit des mêmes sorties
comparées à deux oracles, sans double comptage. Chaque lot utilise un seul
dispatch de produit matriciel, avec réutilisation de poids entre huit tokens.

| Tokens N | Sorties comparées par oracle | Médiane GPU, ms |
|---:|---:|---:|
| 8 | 12 288 | 2.086500 |
| 16 | 24 576 | 3.964875 |
| 32 | 49 152 | 6.291875 |
| 64 | 98 304 | 3.380917 |
| 128 | 196 608 | 6.400958 |

## Q2_K : un expert complet 2048 × 4096

Les **114 688 sorties finies** sont identiques bit à bit à la routine ARM
`nrc=1`, pour N=8/16/32. L'oracle appelle une routine native par sortie ;
son temps n'est pas un témoin de vitesse CPU10. La quantification CPU des
activations Q8_K est hors des horodatages des commandes GPU.

| Tokens N | Sorties comparées | Médiane GPU, ms |
|---:|---:|---:|
| 8 | 16 384 | 3.211625 |
| 16 | 32 768 | 2.168375 |
| 32 | 65 536 | 3.741375 |

Les temps sont ceux de commandes GPU sur données préparées, après un
échauffement, avec trois répétitions. Les variations entre lots ne suffisent
pas à recommander une taille optimale. Aucune comparaison temporelle CPU10
équivalente, lecture SSD de poids GLM, intégration du graphe complet,
comparaison de routes/états KDA ou accélération globale de préfill n'est établie.
L'objectif A reste **non qualifié**.

Les JSON de résultats, reçus d'exécution, journaux du build Q2 et observations
par cas sont sélectionnés ici. Les `.bin`, `.f32`, bibliothèques, exécutables et
poids sont omis. `selection.json` indique pour chaque source sa taille et son
SHA256 originaux, puis ceux de la sélection. Seuls les préfixes de chemins
locaux ont été remplacés par des alias ; les valeurs numériques sont conservées.
`summary.json` est une synthèse dérivée des résultats, avec son périmètre explicite.

Le refus initial sur batterie est conservé séparément dans
[`../metal-components-admission-refusal.json`](../metal-components-admission-refusal.json).
Il précède ces exécutions autorisées et ne les remplace pas.
