# Velocity : protocole de génération et critères de décision

Point de départ : PR #2, `8775b1e4abb729c02079e91c86e2b9a5cf25aded`.
Les options de recherche sont explicites. Aucun changement de précision n'est
assimilé à une optimisation exacte, et aucun numéro de version stable 7.0
n'est attribué à cette campagne.

## Coût à minimiser

Pour une sortie effectivement émise de N tokens, on mesure
`N / (T_construction + T_préremplissage + T_décodage + T_fin_lectures)`.
Le débit hors construction est aussi conservé. L'encodage des prompts et la
vérification cryptographique des fichiers ont leurs propres temps. Cette
dernière réchauffe le cache OS : ces essais ne sont pas un protocole SSD froid.

Un coût de chargement additionne au moins le transport des octets, leur
importation dans MLX et leur conversion éventuelle. Les sommes des temps des
workers peuvent se chevaucher entre elles et avec le calcul : on ne les
soustrait pas naïvement pour inventer un temps GPU. Les timestamps des commandes
Metal ne sont pas instrumentés. `gpu_time_seconds=null` signifie inconnu.

En spéculation de profondeur K, le coût utile est
`(T_draft(K) + T_vérification(K) + T_correction + T_E/S_finales) / tokens_émis`.
Une bonne acceptation ne suffit pas si le draft réexécute tout le préfixe.
Le draft incrémental conserve ses propres KV, tronque les propositions rejetées
et rejoue au plus un token commun pour retrouver le prochain argmax. Avec deux
tokenizers différents, les raccords de réencodage restent vérifiés.

Pour un routage exact avec E experts et k experts activés, une borne analytique
de lecture avant cache est `poids_partagés_utiles + poids_experts_activés`.
La fraction `k/E` des poids experts n'est qu'une approximation de volume pour
des experts de même taille ; ni le débit SSD ni les lectures physiques ne
s'en déduisent. Un objectif de r tokens/s nécessite au moins
`r * octets_physiques_par_token` de service de stockage et assez de calcul.
Les octets logiques ne peuvent pas remplacer ce terme.

## Contrôles expérimentaux

- Vrais prompts et génération gloutonne jusqu'à EOS ou au plafond ; aucun token
  futur fourni à l'algorithme pendant la mesure de génération.
- Six prompts courts de domaines et langues différents, puis un contexte plus
  long et des plafonds 64/128 lorsque les contrôles précédents passent.
- Répétitions et ordre mélangés par graine fixe. Chaque variante est appariée au
  contrôle du même prompt et de la même répétition.
- Poids et tokenizers immuables vérifiés par SHA256. Sources utilisées hashées ;
  toute mutation détectée pendant un worker invalide sa campagne primaire.
- Même somme des budgets de poids, E/S, KV et espace de travail. Avec un draft,
  les deux partitions sont soustraites au budget de référence. Le plafond global
  de l'allocateur MLX est rétabli à leur somme.
- Les lectures asynchrones des deux modèles sont terminées avant la fin du
  chronomètre et les compteurs finaux. Leur coût ne disparaît pas dans close().
- Les ratios des options qui activent le préchargement se comparent également
  à `prefetch`, pour distinguer un gain nouveau du mécanisme déjà présent.
- RSS et réserve système échantillonnés par un processus externe ; swap relevé.
  Les bornes de file et les pics de cache déclarés ne sont pas des mesures RSS.
- Les imports sans copie peuvent vivre hors allocateur MLX : les budgets de
  cache/E/S et le RSS restent nécessaires, même avec un petit pic MLX déclaré.

## Mémoire et stockage du grand modèle

Un fichier partiel, une taille préallouée ou quelques experts lus ne constituent
pas un checkpoint complet. Avant le premier token : vérifier tous les shards,
les métadonnées et les empreintes, arrêter le téléchargement, puis utiliser un
worker borné avec surveillance externe. Commencer par un token, puis quelques
tokens avant 32, 64 et 128. Garder la réserve macOS et le cap de swap.

Le checkpoint Qwen retenu réutilise sa quantification affine existante. Une
parité contre ses poids déquantifiés valide le backend pour ce checkpoint,
pas l'absence de dégradation par rapport au modèle BF16 source. Une fixture
minuscule ne valide pas la génération du checkpoint complet.

## Antériorités

La spéculation et la gestion de KV sont des techniques connues :
[Leviathan et al.](https://arxiv.org/abs/2211.17192).
Le regroupement et l'exécution asynchrone utilisent les APIs de
[MLX](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).
L'import DLPack partage les buffers conformément à l'implémentation officielle
[MLX v0.32.4](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/python/src/convert.cpp).
Aucune invention de ces principes n'est revendiquée.

[SpecMD, Apple, mai 2026](https://machinelearning.apple.com/research/specmd-expert-prefetching)
étudie l'interaction entre caches d'experts et matériel. Ses gains publiés
appartiennent à son protocole ; ils ne sont pas des mesures AirLLM sur M5.
La priorité de cette campagne reste l'effet sur le coût complet des sorties.
