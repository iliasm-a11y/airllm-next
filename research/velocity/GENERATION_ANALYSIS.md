# Analyse indépendante de la génération Granite

Source : `evidence-velocity/generation-metal.json` (90 générations, six prompts, trois répétitions, cinq variantes). Aucune exécution exclue. Les listes de tokens sont identiques dans les 18 comparaisons appariées de chaque variante.

Les 38 empreintes de code du rapport correspondent toutes au commit `574feec`. Le rapport enregistre `8775b1e` comme HEAD au lancement : les modifications mesurées ont été commitées ensuite. Son contrôle de stabilité pendant la campagne est vrai.

Chaque variante émet 507 tokens : neuf pour le cas 0 arrêté sur EOS, 32 pour chacun des cinq autres cas et chaque répétition. Le débit agrégé divise ces 507 tokens par la somme du temps de génération, préfill et drain des lectures inclus. Le chargement initial est séparé.

| Variante | Débit agrégé, tok/s | Gain vs référence | Gain vs prélecture | CPU, s | Chargements |
|---|---:|---:|---:|---:|---:|
| reference | 4.807 | 1.000× | 0.842× | 81.70 | 262,607 |
| prefetch | 5.711 | 1.188× | 1.000× | 89.83 | 262,922 |
| queued8 | 5.771 | 1.201× | 1.011× | 95.74 | 263,492 |
| zero-copy | 6.166 | 1.283× | 1.080× | 80.96 | 262,360 |
| queued4-zero | 5.803 | 1.207× | 1.016× | 89.27 | 262,394 |

Le gain supplémentaire observé de `zero-copy` au-delà de la prélecture est 1.080× (+8.0% de débit ; 7.4% de temps en moins). Il gagne 18/18 couples. Le ratio médian est 1.079× ; moyenne 1.079× ; étendue 1.022–1.145×.

IC bootstrap exploratoire à 95% pour le ratio agrégé `zero-copy`/prélecture : [1.065 ; 1.095] en rééchantillonnant les 18 couples ; [1.068 ; 1.091] en rééchantillonnant les six prompts avec leurs trois répétitions. Ces intervalles portent sur cette petite sélection et cette session ; ils ne garantissent pas un gain sur un autre modèle ou sur des prompts nouveaux.

Résultats négatifs : `queued8` ne gagne que 1.011× au-delà de la prélecture (13/18 couples plus rapides). Son IC apparié [0.999 ; 1.023] comprend 1 ; l’IC par prompt [1.0002 ; 1.0208] l’effleure. Ces données ne justifient pas un gain robuste de la file seule. Ajouter la file de quatre à `zero-copy` ralentit la campagne : ratio 0.941×, IC par prompt [0.931 ; 0.954], 17/18 couples plus lents. Aucun contrôle `queued4` seul ne figure dans cette campagne ; une interaction factorielle complète ne peut donc pas être estimée.

| Cas | Tokens entrée / sortie | `zero-copy` vs référence, médiane (min–max) | `zero-copy` vs prélecture, médiane (min–max) | File de 4 vs `zero-copy`, médiane |
|---|---:|---:|---:|---:|
| 0 | 20 / 9 | 1.306 (1.220–1.353) | 1.086 (1.022–1.104) | 0.943 |
| 1 | 22 / 32 | 1.271 (1.262–1.305) | 1.075 (1.068–1.092) | 0.924 |
| 2 | 26 / 32 | 1.285 (1.273–1.496) | 1.077 (1.057–1.145) | 0.927 |
| 3 | 21 / 32 | 1.232 (1.232–1.271) | 1.046 (1.041–1.092) | 0.976 |
| 4 | 23 / 32 | 1.271 (1.191–1.282) | 1.077 (1.030–1.112) | 0.944 |
| 5 | 18 / 32 | 1.292 (1.256–1.327) | 1.091 (1.080–1.129) | 0.935 |

`zero-copy` réalise 262,360 imports directs sans repli déclaré. Les chargements restent proches : 262,607 référence, 262,922 prélecture, 262,360 imports directs. Le CPU cumulé diminue de 9.9% par rapport à la prélecture (89.83 → 80.96 s). Cela concorde avec la réduction des copies ; aucune mesure ne sépare le temps GPU.

| Variante | Lectures logiques, GiB | Stockage attribué au processus, MiB | RSS après génération, min–max MiB |
|---|---:|---:|---:|
| reference | 383.15 | 50.28 | 550.9–606.1 |
| prefetch | 386.53 | 43.09 | 581.6–608.6 |
| queued8 | 386.07 | 109.55 | 549.1–604.1 |
| zero-copy | 387.19 | 114.70 | 579.6–605.7 |
| queued4-zero | 385.40 | 14.47 | 580.7–603.9 |

Le pic RSS de toute la durée du processus est 628.0 MiB. Les capacités de cache effectives restent à 256 MiB dans les 90 exécutions. Aucune hausse de swap entre les deux mesures encadrant une génération ; une baisse de 8 MiB existe sur une exécution de prélecture. Les points RSS par variante et les prélèvements intermittents ne constituent pas des pics résidents isolés.

Les lectures logiques comptent les octets demandés au lecteur. Le compteur de stockage est celui du processus et peut inclure des lectures annexes ou de la réadahead ; il ne mesure ni les octets NAND ni une bande passante SSD. Les empreintes SHA ont réchauffé les fichiers et le cache OS n’a pas été purgé. Les faibles octets attribués au stockage ne permettent pas d’attribuer le gain à une amélioration des transferts SSD.

Le checkpoint pèse 2.669 Go (2.486 GiB) et tient dans les 16 GiB physiques. Ce résultat ne valide pas un modèle plus grand que la RAM, ni 70B/GLM/Qwen, ni un SSD externe, ni une charge thermique longue. Les sorties ne sont pas évaluées pour leur qualité : le cas 1 appelle notamment « Fibonacci » la suite de comptage proposée. Température, énergie et temps GPU sont absents.

Les comparaisons supplémentaires, toutes les distributions par cas, les compteurs et les méthodes bootstrap figurent dans `evidence-velocity/generation-analysis.json`. Les 20 000 tirages appariés utilisent la graine 20261023 ; le bootstrap par prompt énumère exactement les 46 656 rééchantillonnages possibles. Aucune correction pour comparaisons multiples ou sélection du meilleur résultat n’est appliquée.
