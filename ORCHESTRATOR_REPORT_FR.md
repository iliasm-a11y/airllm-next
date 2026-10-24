# Orchestrateur multi-agents local — résultats et limites

Mesures du 24 octobre 2026, MacBook Air Apple M5, 16 Gio, Metal, SSD interne,
Python 3.12.13, macOS 27.2. Branche de recherche issue de
`ca3a42935be876bbdbbc1e2adbd16c8547746175` (PR #8), indépendante de RONIN.

## Résultat établi

L'orchestrateur exécute les charges réelles SmolLM2 aux paliers **1, 10, 100,
500 et 600 agents logiques**, autour d'une seule instance de modèle et avec
**une seule inférence simultanée**. Les 2 422 tokens des 1 211 travaux de la
campagne finale sont identiques aux trois références ordinaires cycliques.
La réserve reste protégée, la pression observée reste normale et aucune
croissance du swap n'est observée. Le premier essai complet est aussi conservé
séparément dans les preuves ; les chiffres ci-dessous proviennent du code final.

## Implémentation

- Registre indépendant des travaux : enregistrés, actifs, en attente, suspendus,
  travaux par état, inférences réellement en cours et maximum observé.
- Backend partagé unique, capacité déclarée explicite, adaptation au nombre de
  lanes autorisées et aux réservations mémoire ; backend natif limité à 1.
- Priorités avec vieillissement, rotation par quantum, quotas cumulatifs par
  agent pour travaux et tokens de sortie réservés ; capacités globales bornées.
- Budgets distincts pour registre/transcripts, KV, état disque et écritures.
  Aucun tableau KV par agent inactif ; reconstruction depuis les tokens et
  vérification exacte du rejeu. Les plafonds natifs du modèle restent appliqués.
- Suspension et reprise persistantes, verrou exclusif, checksum, fsync et
  remplacement atomique. Un redémarrage récupère les travaux suspendus.
- Outils fichier explicitement accordés dans les deux registres, espaces par
  agent, quotas persistants, processus à environnement nettoyé, délai maximal,
  plafonds de fichiers/entrées/sorties et deux workers globaux maximum.
- Métriques de file, premier token, service, fin, débit, RSS, réserve et swap ;
  arrêt coopératif sous pression et supervision externe du groupe possédé.

Les interfaces et calculs existants ne sont pas réécrits : seuls des modules
facultatifs et des exports différés sont ajoutés, avec des liens documentaires.
Le [guide API et protocole](research/orchestrator/README.md) précise les contrats.

## Charge réelle SmolLM2-135M-Instruct

Deux tokens demandés par agent, trois prompts cycliques, priorités 0/1/2,
quantum de deux tokens, journaux réellement écrits. Temps total = création des
identités + admission + drain ; chargement du modèle et références exclus.
La TTFT inclut l'attente depuis la soumission. RSS échantillonné par ordonnanceur.

| Agents | Total (s) | Tokens/s de charge | TTFT médiane / p95 (s) | RSS pic (Mio) | Inférences simultanées max |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.211 | 9.491 | 0.154 / 0.154 | 440.16 | 1 |
| 10 | 2.095 | 9.547 | 1.085 / 1.825 | 440.30 | 1 |
| 100 | 21.440 | 9.328 | 10.674 / 20.222 | 442.11 | 1 |
| 500 | 113.460 | 8.814 | 56.841 / 106.214 | 443.66 | 1 |
| 600 | 137.484 | 8.728 | 69.245 / 128.234 | 451.28 | 1 |

Processus complet supervisé : **277.59 s**, pic RSS
**455.11 Mio**, réserve disponible minimale
**5.33 Gio**, swap initial/final nul et
croissance observée nulle. Le compteur du guard porte sur le processus et ses
descendants ; le compteur de l'ordonnanceur porte sur son processus propriétaire.
Les durées sont des observations courtes sur un hôte non isolé, sans intervalle
de confiance ni certification thermique.

Le checkpoint entraîné est petit (~269 Mo de poids) et tient dans la RAM.
SHA-256 final de `model.safetensors` :
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
Les fragments textuels et tous les IDs sont conservés dans le JSON brut.

## Ordonnanceur synthétique

Backend déterministe avec délai de service de 1 ms et quatre appels autorisés,
sans modèle entraîné ni calcul MLX. Ces durées mesurent registre, files,
adaptation, journaux et appels simulés. Leur débit ne décrit aucun modèle.

| Agents | Total (s) | Appels synthétiques simultanés max | RSS pic (Mio) |
|---:|---:|---:|---:|
| 1 | 0.0032 | 1 | 25.83 |
| 10 | 0.0149 | 2 | 26.03 |
| 100 | 0.1661 | 4 | 26.69 |
| 500 | 2.3926 | 4 | 29.12 |
| 600 | 3.3207 | 4 | 32.23 |
| 1000 | 8.7895 | 4 | 37.72 |

Le palier 1 000 confirme l'absence de plafond codé à 600. Les limites par défaut
sont configurables et restent assujetties aux budgets ; aucun maximum universel
n'est établi. La TTFT synthétique est une borne supérieure du premier quantum.

## Contrôle Qwen GGUF et reprise réelle

Le Qwen3.6-35B-A3B GGUF de **23 938 321 664 octets** est réellement exécuté :
un agent, deux tokens, sortie brute `4<|im_end|>`, identique à `model.generate()`.
Charge : **6.716 s** ; processus complet **16.06 s**,
pic RSS supervisé **323.12 Mio**,
une inférence, aucune croissance du swap observée. Ce contrôle valide le chemin
GGUF de l'adaptateur ; il n'est pas une charge Qwen à 600 agents.
Le hash intégral du fichier Qwen est recalculé après les essais et correspond à
`f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d` ;
les deux vérifications de poids sont conservées dans `checkpoints.json`.

Deux workers SmolLM2 distincts vérifient aussi la reprise : le premier engage
un token de chacun de deux travaux et suspend ; après sortie complète de ce
processus, le second rouvre l'état, reconstruit les contextes, alterne les quanta
et termine quatre tokens par travail. Les huit tokens finaux sont identiques
aux références ordinaires. Les deux guard reports sont sans violation.

Deux pilotes Qwen échoués sont conservés : budget de métadonnées du tokenizer
insuffisant, puis absence de `fingerprint` dans le lecteur GGUF. Le chargeur
reprend désormais le contrat déjà validé dans `throughput_gguf.py` (lecteur
256 Mio, métadonnées 192 Mio, avant création du modèle), à l'intérieur du même
payload supervisé de 768 Mio. L'adaptateur GGUF utilise un hash d'index et les
métadonnées du descripteur, contrôlées avant/après travail. Un test rejette les
mutations. **Les plafonds RSS/réserve/swap n'ont pas été abaissés.**

## Validation et preuves

**1 462 tests réussis, 1 test ignoré (compression CUDA/bitsandbytes sans NVIDIA),
20 sous-tests réussis**, via `scripts/validate_throughput.sh`, sous supervision.
Le protocole hérité exclut `air_llm/tests/test_streaming_gpu.py`. Les lectures
réelles GGUF et la capture locale historique du routeur sont activées ; cette
capture et les poids restent exclus de la publication. Les 64 nouveaux tests
couvrent les cinq paliers, reprise/identité, quotas, priorité/vieillissement,
concurrence, pression/swap/RSS simulés, journal/libération en échec et outils.
Compilation Python et `git diff --check` passent.

Les JSON bruts, logs, guard reports, XML de tests, reprise, pilote échoué et
première campagne sont dans `research/orchestrator/evidence.tar.gz`.
`research/orchestrator/evidence-manifest.json` fournit tailles et SHA-256 de
chaque fichier. Le CSV de synthèse final est dans
`research/orchestrator/results/summary.csv`. Les empreintes des quatre sources
d'exécution sont vérifiées identiques aux fichiers publiés, avant/après chacun
des trois essais finaux. Le code et les commandes de reproduction restent
accessibles dans le dépôt, sans poids ni dépendance RONIN.

## Limites explicites

1. **600 agents logiques ne signifient pas 600 générations simultanées.** Le
   backend natif testé garde une seule génération à la fois. L'attente croît
   avec la file, même si le registre et la mémoire restent bornés.
2. Les charges réelles utilisent seulement deux tokens et des prompts courts ;
   elles ne valident ni conversations complètes, ni qualité d'assistant,
   ni 600 tâches longues, ni GLM complet. Le contrôle Qwen reste unitaire.
3. Le rejeu libère la mémoire mais peut coûter très cher pour des sorties
   longues et des alternances fréquentes ; aucun gain universel n'est annoncé.
4. Le contrat accepte d'autres backends locaux, mais seule la capacité native
   1 et la simulation de capacité 4 sont vérifiées. Déclarer une capacité ne
   prouve pas que le moteur calcule réellement plusieurs séquences en parallèle.
5. Le moniteur Python coopère entre tokens ; le superviseur externe arrête le
   groupe possédé si un forward dépasse les protections. Les échantillonnages
   ne sont pas des limites d'allocation instantanées. Les tests de pression
   utilisent des échantillons injectés, sans provoquer volontairement de swap.
6. Les outils exposent des capacités fichier limitées, sans shell, réseau ou
   code arbitraire. Ils ne constituent pas une sandbox OS contre du code hostile.
   Le modèle ne déclenche aucun outil automatiquement.
7. Les identités de checkpoint fondées sur métadonnées/stat/index ne sont pas
   une authentification cryptographique de tous les poids. Les journaux et
   répertoires d'état doivent rester locaux et de confiance.
8. Les snapshots complets sont réécrits aux mutations : leurs coûts augmentent
   avec les travaux retenus. Le CSV rapporte admission et drain séparément ;
   1 000 agents synthétiques ne prouvent pas une efficacité à très grande échelle.
9. Cache OS non purgé, SSD interne, pas de mesure NAND/SSD externe, énergie ou
   thermique prolongée. Les compteurs mémoire sont des pics échantillonnés.
10. Recherche en PR brouillon dépendante de la PR #8. Aucune fusion dans `main`,
    aucune version stable ni compatibilité universelle n'est annoncée.
