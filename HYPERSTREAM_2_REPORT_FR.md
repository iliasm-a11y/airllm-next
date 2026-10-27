# AirLLM Next HyperStream 2 — code livré, campagne M5 différée

27 octobre 2026. **Aucun gain de performances HyperStream 2 n'est établi.**
La sélection adaptative, le profilage et la campagne vérifiable sont implémentés.
La comparaison réelle du checkpoint reste à exécuter : le Mac M5 accessible
est resté en pression mémoire d'avertissement pendant la fenêtre de dix minutes.
Cette livraison est une recherche expérimentale en brouillon, pas la validation
de l'objectif final de vitesse.

## Ce qui explique le cas historique à 110,7 secondes

Les preuves originales de la PR #10 ont été relues et leurs sources comparées
au commit `c8f81e57463ed3d7fd3d7dba226a65500439a9ce` : tous les fichiers Python
du moteur concordent exactement. L'analyse recalculée figure dans
`research/hyperstream2/historical-analysis.json` ; le JSON original et sa garde
sont conservés dans la nouvelle archive de preuves.

| Observation historique vérifiée | Valeur |
|---|---:|
| Tokens d'entrée, template compris | 364 |
| Taille des blocs | 16 |
| Passages complets à travers les 40 couches | 23 |
| TTFT moteur | 110,699902 s |
| Premier forward | 110,697301 s |
| Lectures logiques du premier forward | 135 518 514 304 octets |
| Stockage imputé par macOS à ce forward | 145 363 525 632 octets |
| Sortie complète historique | `42`, puis EOS |

L'hypothèse d'un prompt entièrement exécuté token par token est donc réfutée
pour ce cas : le moteur groupait déjà les tokens par 16. En revanche, les
poids et unions d'experts nécessaires sont relus lors des passages successifs
sur de petits blocs. C'est un coût réel qui justifie l'expérience de blocs
plus grands sans changer les experts choisis.

Les compteurs historiques de toute la génération donnent 52,672 s dans la
préparation/lecture packed, 19,810 s dans les produits ordinaires et 15,812 s
dans les produits groupés d'experts. **Ces durées inclusives se recouvrent.**
Elles ne fournissent pas une partition de 110,7 s et ne permettent pas
d'attribuer précisément le solde au GPU, CPU, SSD physique ou aux attentes.
Le nouveau profil détaillé, nécessaire à cette attribution, n'a pas pu tourner.
Les octets de stockage imputés au processus ne sont pas une mesure NAND ; ils
ne sont pas interchangeables avec les octets logiques.

## Changements exécutables

1. `HyperPrefillPolicy` choisit un bloc sous les budgets de travail, de contexte
   et de réserve existants. Le plafond expérimental initial est 128 tokens.
   Les activations, résultats MoE, scores d'attention et états récurrents/KV
   sont estimés ; la moitié du workspace reste disponible pour les tuiles et
   allocations transitoires. Les gardes réelles restent autoritaires et peuvent
   refuser une estimation admise. Le contexte est alors restauré.
2. Le runtime réévalue cette politique à chaque bloc et conserve un token par
   étape de décodage. Aucun poids, expert, routage ou kernel n'est supprimé ou
   remplacé. Aucune quantification ni conversion du checkpoint n'est ajoutée.
   Le backend et le CLI exposent `hyper_prefill` / `--hyper-prefill`, **inactif
   par défaut**. Les options de transport et de pipeline H1 restent séparées.
3. Une trace monotone bornée enregistre spans, phases, événements, compteurs,
   temps exclusifs et unions d'intervalles. L'observateur réversible couvre
   lectures, projections, attention, MoE et attentes MLX déjà présentes, sans
   commande de synchronisation supplémentaire. Les durées sont côté hôte,
   jamais présentées comme des temps Metal purs.
4. Le harnais peut conserver tous les logits FP32 bruts et les experts choisis
   pour chaque position de chaque couche. Le vérificateur indépendant standard
   Python recalcule hashes, égalité des tokens/routes/positions/EOS, erreurs
   numériques, contrôles de ressources et statistiques appariées. Il rejette
   notamment NaN/Inf, routes élaguées, fichiers altérés, options non déclarées,
   protection relâchée et campagne incomplète.
5. Le générateur de campagne prépare dix paires AB/BA, avec processus frais,
   sources et protocole figés. Profilage détaillé et corroboration des temps
   sans observateur sont séparés. La CI Ubuntu exécute les contrôles portables.

Un changement de forme matricielle peut modifier les arrondis. **La fidélité
réelle du bloc 128 n'est pas encore validée.** L'acceptation pré-déclarée exige
routes et tokens exacts, et tous les logits sous `atol=1e-4, rtol=1e-4`. L'égalité
bit à bit est une mesure séparée. Le routage futur n'est pas prédit ; aucun
préchargement spéculatif supplémentaire n'est introduit sans justification.

## Essais actuels et refus de lancement

Matériel vérifié : MacBook Air M5, 16 Gio, Metal disponible, MLX 0.32.4.
Le checkpoint historique est sur le SSD interne. Aucun disque physique externe
n'est énuméré. `xctrace` n'est pas installé. `pmset` ne rapporte pas d'alerte
thermique, ce qui ne mesure ni la température ni un comportement thermique
soutenu.

Les 299 relevés de la fenêtre d'attente montrent tous une pression macOS de
niveau 2, alors que le lancement exige le niveau normal 1. La mémoire disponible
varie de 2 710 962 176 à 4 669 030 400 octets. Le swap système passe de
419 102 720 à 695 336 960 octets pendant cette fenêtre ; ce compteur est global
et aucun worker d'inférence n'a été lancé. Aucune application tierce n'a été
fermée et aucun seuil n'a été abaissé.

Les paramètres conservés pour l'inférence sont : réserve 3 Gio, RSS 2 Gio,
croissance swap 128 Mio, admission 768 Mio, intervalle 50 ms et 32 processus
possédés au maximum. Le calcul d'empreinte préalable, qui a son propre petit
payload déclaré de 64 Mio, a lui aussi été refusé sous pression. Il n'y a donc
**pas de nouveau SHA-256 intégral avant/après ni de nouveau benchmark**. Le SHA
historique reste une preuve historique, sans certification actuelle implicite.

| Mesure demandée | Référence historique | Candidat actuel |
|---|---:|---|
| TTFT pour 364 tokens | 110,70 s | Non mesuré |
| Débit préfill/décodage | Données H1 conservées | Non mesuré |
| Octets lus, RSS, swap par requête | Données H1 conservées | Non mesuré |
| Fidélité complète du checkpoint réel | Contrôles H1 conservés | À valider |
| Objectifs <60 / <30 / <15 / <5 s | — | Aucun atteint démontré |

## Validation et preuves

- **32 tests portables réussis**, sans MLX : trace, observateur, admission du
  planificateur, falsification du vérificateur et protection des fichiers.
- **111 tests existants réussis et 20 sous-tests**, couvrant campagnes,
  profilage et analyse de résultats. Journal et JUnit conservés.
- **20 tests CPU sur miniature réussis, 1 variante GPU non exécutée** : logits
  indépendants, cache/positions, routes, séquences, EOS et restauration après
  erreur. Leur stdout provient de la transcription de l'outil, explicitement
  identifiée dans l'archive ; aucun journal de processus n'avait été enregistré.
- La suite complète historique et les tests GPU n'ont pas été relancés dans
  ces conditions. Les résultats H1 « 1324 passed » ne sont pas réattribués à H2.
- L'archive `evidence-hyperstream2/raw-evidence.tar.gz` a été extraite dans un
  nouveau dossier ; toutes ses empreintes ont été recomparées au manifeste.
  Elle contient les refus et tests réels ainsi que la référence historique,
  **aucune génération H2 fictive** et aucun poids du modèle.

## Résultats acquis, hypothèses et limites

Acquis : analyse historique reproductible, code facultatif, protections
conservées, mécanismes de trace et de vérification testés, contrôles numériques
sur miniature. Réfuté pour le cas historique : absence de tout préfill groupé.
À éprouver : le gain et la fidélité réelle des blocs plus grands, leur meilleur
plafond, la part GPU pure et le comportement thermique prolongé.

La politique est intégrée au GGUF Qwen35 MoE expérimental. Elle ne valide pas
Qwen3.8 BF16, d'autres familles, SSD externe, froid contrôlé, contexte très long
ou modèles de 100 Go à 1 To. Les copies/vues de cache et les estimations de
mémoire doivent être confrontées aux pics MLX/RSS réels. Les protections
globales restent nécessaires. Aucun mécanisme supplémentaire de fusion Metal,
cache prédictif ou transfert direct SSD→GPU n'est annoncé comme réalisé.

Le travail continue depuis une branche dédiée empilée sur HyperStream 1 ;
`main` et la PR #10 ne sont pas fusionnés ou modifiés. Le protocole exact de
reprise figure dans `research/hyperstream2/README.md`. La mission de gain réel
reste inachevée tant que la campagne matérielle n'a pas fourni de résultats
numériquement acceptés et corroborés sans instrumentation détaillée.
