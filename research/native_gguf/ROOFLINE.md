# Roofline du chemin GGUF natif — analyse des preuves existantes

Source : `evidence-native-gguf/inspect.json`, index du fichier de 23 938 321 664 octets et 1 194 tenseurs. Cette analyse ne lit aucun poids et ne lance aucun benchmark. Les octets indiqués sont des tailles encodées GGUF et les appels des projections exactes du code courant.

## Octets nécessaires pour une position isolée sans cache

| Famille réellement exécutée | Octets encodés par position |
|---|---:|
| Poids partagés des 40 couches, routeurs et petits paramètres | 894,773,760 |
| Experts sélectionnés : 8 / 256 dans chacune des 40 couches | 652,738,560 |
| Projection finale complète de 248 320 lignes | 417,177,600 |
| Norme finale | 8,192 |
| Une ligne de token embedding | 1,152 |
| **Total froid sans cache** | **1,964,699,264** |

Les tenseurs vision (`v.*`) et MTP (`mtp.*`) sont exclus : ils ne sont pas exécutés par ce chemin texte. Une ligne de l’embedding suffit ; sa table complète ne doit pas entrer dans le volume par position.

## Taille des tuiles et nombre de lancements

| Lignes par tuile | Experts actifs | Blocs partagés | Sortie | Total / position |
|---:|---:|---:|---:|---:|
| 512 | 1920 | 1360 | 485 | 3765 |
| 2048 | 960 | 460 | 122 | 1542 |
| 4096 | 960 | 350 | 61 | 1371 |

Passer de 512 à 2 048 lignes réduit le compte théorique de 3 765 à 1 542 appels par position, soit 59,04 %. Les octets mathématiquement nécessaires et le routage restent identiques. Le débit réel doit être mesuré : cette baisse du nombre d’appels ne démontre pas une accélération du même facteur. La ligne 4 096 est uniquement une projection arithmétique.

## Lecture des mesures disponibles

| Preuve | État | Positions | Lignes | Octets / position | Appels / position | Débit global |
|---|---|---:|---:|---:|---:|---:|
| `first-complete-metal.json` | completed | 1 | 512 | 1,964,699,264 | 3,765 | 0.4002 tok/s |
| `short16-metal.json` | completed | 16 | 2048 | 1,952,808,192 | 1,542 | 0.9566 tok/s |
| `short4-metal.json` | completed | 4 | 512 | 1,917,235,328 | 3,765 | 0.4626 tok/s |
| `stage32-metal.json` | completed | 32 | 2048 | 1,958,753,728 | 1,542 | 0.9640 tok/s |

La série courte à 512 lignes économise 47,463,936 octets par position par rapport au parcours froid, soit 2.42 %. Son taux de hits est 2.80 %. Les 15 060 appels observés correspondent exactement à quatre parcours de 3 765 appels : les hits de poids évitent certaines lectures, mais aucune multiplication.

Dans cette série, `packed_read_seconds` vaut 1,281 s sur 8,647 s de génération (14,81 %) ; `matmul_seconds` vaut 5,381 s (62,23 %). Le second compteur couvre les imports MLX, lancements, affectations et synchronisations ; il ne mesure pas le temps GPU isolé. Le cache conserve des tableaux NumPy compactés : `packed_matmul` recrée une entrée MLX à chaque appel, y compris après un hit. Le compteur de stockage du processus conserve sa qualification OS et ne certifie pas les octets NAND.

## Cache : constat et prochain essai borné

Les clés de tuiles sont distinctes par nom, borne de lignes, représentation compactée et expert. Les clés expert sont des tuples `(packed:nom@début:compte, expert)` ; le classificateur existant les reconnaît déjà. Les hints de taille du chemin compacté valent bien `compte * row_bytes`. Aucun correctif de clé ou de taille n’est justifié par cette lecture.

Le cache adaptatif existant a une demi-vie de fréquence de 1 024 requêtes et une histoire non résidente de 4 096 clés. À 512 lignes, une position traverse 3 582 demandes cachables dans la série courte ; à 2 048 lignes, les séries 16/32 positions en traversent 1 722. Les changements d’experts ajoutent des clés et peuvent raccourcir la mémoire effective de l’historique. Les compteurs cumulés ne permettent pas d’attribuer chaque miss à une éviction d’historique.

Les séries de 16 et 32 positions montrent exactement **203 hits** et **190 257 152 octets économisés** chacune. Les 16 positions supplémentaires n’ajoutent donc aucun hit. Le cache reste rempli à 266 Mo (environ 254 Mio) : la capacité seule n’explique pas une absence totale de réutilisation de paramètres partagés.

Une cause structurelle plausible est le score existant `max(0, fréquence_décroissante - 1) * coût`. Pour une clé demandée une fois toutes les R requêtes avec une demi-vie D, son régime périodique idéal vaut `F = 1 / (1 - 2^(-R/D))` juste après utilisation. Son bénéfice devient nul après `D * log2(F)` requêtes. Avec R=1 722 et D=1 024, F vaut environ 1,45 et le bénéfice tombe à zéro après environ 552 requêtes, avant la prochaine réutilisation. Avec R=3 582, la fenêtre positive est d’environ 137 requêtes. Ce calcul explique un mécanisme possible de rotation des résidents ; il ne remplace pas une trace individuelle des évictions.

La projection finale vaut 417 177 600 octets, contre 268 435 456 octets pour tout le cache. Elle ne peut pas résider intégralement. `output_tile_cache=false` évite que son balayage entre en compétition avec les poids réutilisés. On peut retenir une fraction de ses tuiles, mais cela consomme le même budget que les autres poids partagés et ne diminue pas le plancher global.

Même un cache idéal de 256 Mio ne peut éviter plus de 13.66 % des octets d’une position isolée en régime établi, avec le chemin de calcul actuel : plancher 1,696,263,808 octets / position. Ce plancher porte sur les lectures de poids de l’application, sans annoncer de débit SSD ou de durée.

Premier essai ciblé recommandé : conserver la politique adaptive et utiliser son paramètre existant `decay_interval=32768` face à 1 024, avec exactement le même budget de 256 Mio, les mêmes tuiles de 2 048 lignes, mêmes IDs d’entrée et limite de génération. Cette ablation change une seule constante de l’heuristique existante. Comparer IDs/logits, latences hors première position, octets, hits/misses et pic RSS. Elle teste le mécanisme décrit ci-dessus ; son débit supérieur n’est pas établi.

L’option existante `policy=shared` permet ensuite un essai indépendant de préférence de classe, avec les mêmes limites. La préférence shared est justifiée comme hypothèse par la réutilisation systématique des blocs, mais un balayage partagé de 894,8 Mo dépasse toujours le cache de 256 Mio ; la préférence seule peut donc continuer à tourner. Ne pas modifier simultanément la capacité, les mathématiques, le nombre d’experts ou le mode I/O.

Une priorité future pourrait conserver les tuiles compactées chaudes directement en MLX sous le même budget. Cela demanderait une mesure séparée des allocations Metal et des copies évitées ; ce changement n’est ni implémenté ni crédité ici.

Les résultats et hashes des sources lus sont enregistrés dans `evidence-native-gguf/roofline.json`. Les séries historiques ont des hashes GGUF-Qwen différents du fichier courant ; leurs chiffres restent attachés aux hashes enregistrés dans chacune de leurs preuves. Des séries de longueurs différentes, avec cache OS non contrôlé, ne constituent pas une ablation causale. Un run encore en cours conserve son état et ne doit pas être présenté comme une mesure finale.
