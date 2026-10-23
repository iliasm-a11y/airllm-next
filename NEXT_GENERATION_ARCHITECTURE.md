# Architecture AirLLM Next 5.0.0a1

Base exacte : `9629c27b9d8bdf697ae866498ad468e11eb183d4` (4.1.0rc1).
Branche : `research/airllm-next-5-generation`. Le moteur existant, ses
backends, les packs réversibles et l'historique expérimental sont conservés.

## Hiérarchie de poids intégrée

`NativeWeights` dessert les trois backends depuis leurs vrais sélecteurs de
tenseurs, lignes et experts. La nouvelle réserve `compact_cache_bytes` est
prélevée dans `cache_bytes`, jamais ajoutée au budget :

`cache décodé <= cache_bytes - compact_cache_bytes`

`cache source <= compact_cache_bytes`

Le cache source LRU conserve les octets FP8 et leurs échelles, les mots BF16,
ou les tableaux flottants lus. Il ne requantifie pas. Une vue de lecture
regroupée est facturée par son allocation sous-jacente entière ; des vues
partageant une allocation dans une entrée sont dédupliquées. L'admission ne
recopie pas une grande matrice. Un hit évite le lecteur de checkpoint, puis
réexécute le décodage dans le même chemin numérique. Les sidecars affines
conservent leur traitement distinct.

Le cache de tuiles de sortie est une seconde option indépendante, dans le
cache décodé existant. Il évite de relire une tête vocabulaire à chaque passe
lorsque la capacité et la politique d'admission la conservent. Les identités
incluent le tenseur et les bornes de lignes. Aucun tenseur complet n'est créé
pour bénéficier du tuilage. Une faible localité peut rendre cette admission
défavorable aux autres matrices ; l'option n'est donc pas imposée par défaut.

Les lectures futures restent limitées à `SingleFlight` et son budget distinct.
Les sélecteurs sont validés même lors d'un hit. Une lecture interrompue ne
publie aucune entrée incomplète. La pression mémoire réduit les deux caches,
évacue le préchargement et conserve la réserve système. Les compteurs de
fichiers lus, octets logiques et réutilisation du cache restent distincts des
lectures physiques du système.

## Vérificateur et décision de profondeur

Le vérificateur GLM/Llama conserve son invariant : le cache contient le
préfixe engagé sauf le dernier token en attente. Les experts sont regroupés
sur les positions effectivement calculées dans une passe ; aucune route
future inconnue n'est utilisée comme vérité. Les corrections et bonus sont
produits par le modèle cible. Les positions rejetées sont tronquées.

La politique V5 observe le coût réel brouillon + vérification par token émis,
les octets cible et les octets brouillon lorsqu'ils sont disponibles. Des
passes singleton recalibrent la référence au cours de la génération. Les
mesures par profondeur orientent les sondes et la réduction/désactivation du
travail qui ne s'amortit pas. La métrique d'acceptation seule ne suffit plus.
Les détails et contre-exemples exécutés sont dans
`research/v5/SPECULATION.md` et `evidence-v5/`.

L'exactitude de transport ne garantit pas l'invariance des arrondis entre
singleton et bloc. La spéculation reste gloutonne et optionnelle. Qwen
récurrent conserve son refus explicite de cette voie tant que son rollback
de vérification complet n'est pas intégré.

La validation d'installation a aussi révélé un défaut hérité distinct : des
scores DSA exactement égaux à la frontière du top-k peuvent sélectionner des
clés différentes dans MLX et PyTorch. Ce choix change l'attention, même si les
scores sont égaux. `dsa_topk` conserve `argpartition` pour les cas non ambigus,
mais détecte une égalité finie traversant la frontière et résout les seules
lignes concernées avec le `topk` CPU PyTorch installé. Les masques causaux
restent appliqués ; des égalités de seuls éléments futurs à `-inf` ne déclenchent
pas cette voie.

Ce choix définit une convention de référence CPU locale ; il ne promet pas
le même départage que CUDA, un autre build PyTorch ou tous les kernels serveur.
Le contrôle ajoute une synchronisation explicite lorsque T dépasse k. Les
copies et le scratch top-k sont réservés avant allocation dans le workspace ;
vérification, copie, import éventuel, top-k CPU et replis sont instrumentés
séparément. L'import tardif de Torch et ses métadonnées peuvent augmenter le
RSS au-delà des seuls buffers gérés. Aucun gain Metal n'est attribué à ce fix.

## Partage entre agents

`AgentEngine` conserve un seul modèle et un seul contexte natif résident.
Une file bornée choisit les priorités puis alterne les demandes de même
priorité aux frontières de quantum. Chaque agent possède son propre
transcript. Au changement de demande, le contexte précédent est libéré ; un
rejeu du préremplissage original puis des tokens singleton reconstruit l'état
et vérifie chaque token déjà engagé. Ce compromis évite de multiplier les
KV/états récurrents par le nombre d'agents et peut augmenter la latence.

Les journaux atomiques stockent les tokens, pas les matrices de KV. Leurs
limites de capacité et d'écriture sont explicites. L'identité de reprise lie
checkpoint, backend et options numériques. L'API jobs locale s'exécute sur le
thread propriétaire MLX et permet interrogation, annulation et reprise.
Une annulation attend la fin du forward en cours. Les outils et les missions
externes restent la responsabilité de l'orchestrateur RONIN.

## Admission, pics et périmètre

`preflight` lit les métadonnées et fait un inventaire des fichiers déclarés.
Il sépare reconnaissance d'architecture, format décodable, budget d'état,
présence des fichiers et validation de génération. Les réservations incluent
les snapshots Qwen et la croissance des banques KV. Un précontrôle positif
n'est pas une vérification du contenu des poids ni une borne complète du RSS.

Le cache latent GLM compte désormais l'ancien et le nouveau tampon pendant
un agrandissement, puis matérialise la copie avant de relâcher l'ancien
graphe. La taille finale seule sous-estimait le pic hérité de V4.1.

Les opérations MLX natives restent le calcul de référence. Aucun gain Metal,
NEON ou kernel FP8/NVFP4 spécialisé n'est déclaré sans mesure sur le matériel
correspondant. Les métadonnées Python, les temporaires de déquantification,
l'allocateur, le tokenizer et le cache de pages OS nécessitent toujours des
mesures système en plus des réservations gérées.
