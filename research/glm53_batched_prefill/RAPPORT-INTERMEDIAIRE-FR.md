# RONIN M5 — résultats clos et validation en cours

État du 3 novembre 2026. Les objectifs A (GPU complet), B (cache exact utile) et C (deux agents utiles jusqu’à EOS) restent indépendants et non qualifiés à cette étape. Les composants synthétiques, les captures de fidélité et les tâches fonctionnelles ne sont pas interchangeables.

## Résultats matériels clos

| Essai | Résultat et portée |
|---|---|
| RONIN fonctionnel CPU10/GPU0, batch8, Max | Agent 1 : 522 tokens, préfill 1238,015911 s, 171 IDs puis délai 2700 s ; aucun EOS ni réponse finale utile. Agent 600 et retour1 non lancés. |
| Témoin natif CPU8 sans cache | Trois prompts gelés 1 → 600 → 1 : 521/542/521 tokens, préfills 1285,763678 / 1322,539045 / 1314,183277 s ; quatre vrais échantillons greedy diagnostiques par requête. Code 0, secteur/intégrité validés ; aucune qualification fonctionnelle EOS. |
| Rejeu numérique du témoin | Agent 1 request0 contre2 : 154 880 logits finis, état sérialisé, captures, 15 642 écritures observées, continuations et post-état exacts. Rejeu intra-bras, pas comparaison indépendante de variantes. |
| Préfixe commun | 600 identités natives Max, 211 IDs communs. Trois sérialisations complètes à208 identiques : 156 739 588 octets (149,478519 Mio). Première requête après restauration exacte sur logits/état/continuation ; bras complet encore en cours. |
| Q8 Metal représentatif | M1536/K4096, N8/16/32/64/128 : 380 928 sorties bit à bit face à ARM nrc2 et au graphe CPU10. Poids/activations synthétiques, pas un préfill GLM complet. |
| Q2_K Metal représentatif | M2048/K4096, N8/16/32 : 114 688 sorties bit à bit face au dot ARM nrc1. Poids synthétiques, quantification d’activation CPU native. Pas de routes/logits/états de modèle mesurés. |

Les tolérances Metal existantes sont aussi satisfaites sur ces composants. Elles ne sont pas élargies. Les premiers petits tests Q8 M10/14/32 avaient montré des différences avec certains chemins du dispatcherCPU : celles-ci ne constituent pas à elles seules un rejet global GPU. Aucune mesure de composant n’établit un gain CPU/GPU sur les prompts complets. Les poids préparés sont chauds ; aucune accélération du SSD n’en est déduite.

## Ressources et provenance

Les deux campagnes GLM closes respectent les gardes de ressources observées : RSS échantillonnée maximale6029656064 octets pour la tentative fonctionnelle, 5 608 177 664 pour le témoin natif ; croissance de swap observée0 dans les deux cas. Les alertes de pression 2 sont conservées et admises seulement dans le domaine contextuel du watchdog existant. Cela ne signifie pas que la pression est restée normale. Les limites 8, 5 Gio/32 Mio, les réserves, contrôles secteur/couvercle/horloges, intégrité et identitéPID ne sont pas désactivés.

Le verrou UID du premier témoin natif a été acquis tardivement et n’a pas été transmis rétroactivement au helper. Cette limite est conservée dans son reçu. Le nouveau superviseur acquiert la propriété avant lancement et transmet son descripteur ; ses tests synthétiques couvrent également split. CPU16 et restore8 ont en plus une observation OS du descripteur réel dans le helper.

Le modèle original Q2_K_S reste externe et inchangé. Le runtimeV4 hérité est épinglé à llama.cpp 3f8af15a76ac8675812e856da45454bafc90f12c avec16 patches. Une reconstruction de sources isolée a retrouvé l’ordre et les16 SHA des patches, puis les13 empreintes finales du build. Aucun runtime natif n’a été reconstruit pour cette vérification. Les bibliothèques effectivement mesurées et les en-têtes du helper restent identifiés séparément ; une copie de sources ne constitue pas une validation d’un nouveau binaire.

## CPU16 rejeté et cache CPU8 en cours

L’alimentation secteur a été rétablie après le refus sur batterie. Le premier prompt CPU16 de 521 tokens se termine en 820,050836167 s mais échoue au critère CPU bit à bit : 154 880 logits différents, écart maximal 1,61790836 et état final différent. Les quatre tokens greedy sont identiques. Le contrôleur a donc arrêté le bras pendant la requête suivante ; le wrapper reste `cancelled`, le natif `-15` et le superviseur `failed`, sans violation mémoire. Aucun gain n’est retenu.

Les contrôles précoces 32/128 tokens et les checkpoints 128/208/256/384/512 étaient exacts. La première divergence capturée est au token 512, KDA couche 0, avec un écart maximal de 9,31e-10 ; des routes ordonnées changent couche 4, puis l’ensemble d’experts change au token 513. La queue 9 contre 8+1 explique une hypothèse forte de changement de chemin nrc1/nrc2 ; elle reste à vérifier par une correction isolée et de nouvelles captures complètes. Les seuils ne sont pas élargis. [Preuves du rejet](evidence/cpu16-rejected/README.md).

La qualification du cache poursuit indépendamment le chemin CPU8 inchangé. Les 208 tokens communs ont été calculés une fois en 449,387450125 s, puis le snapshot complet de 156 739 588 octets a été capturé en 0,061756958 s au total. Son SHA correspond aux trois checkpoints de référence. La première lecture native prend 0,004909917 s, et le reset/restauration/vérification complet 0,067238 s. Pour la première requête terminée, les fichiers bruts des 154 880 logits, de l’état final, des quatre vecteurs de continuation et du post-état sont exacts. Ce contrôle partiel n’est pas la comparaison complète des captures, écritures et trois requêtes : le bras reste en cours et non qualifié.

Le snapshot reste unique, immuable, complet (`NONE`) et limité à 256 Mio ; scratch, états actifs et captures sont budgétés séparément. La séquence native ne sauvegarde ni sampler ni logits : suffixe privé non vide et sampler neuf sont obligatoires. `PARTIAL_ONLY` n’est pas une base autonome après reset.

Pour 32+, les frontières du helper `split` à 208 diffèrent de `full` et certains checkpoints requis ne sont jamais atteints. [L’audit statique](split32_protocol_audit.md) décrit ce blocage structurel. L’admission de `split` par le lanceur ne corrige pas le protocole numérique. Aucun essai long de ce chemin ne doit être lancé comme s’il était qualifiable en l’état.

## Verdicts aux douze questions

1. GPU plus rapide sur GLM complet : non démontré ; composants multi-token exacts seulement.
2. Meilleur batch : inconnu ; CPU8 témoin clos ; CPU16 original rejeté, correction non qualifiée.
3. Économie de préfill : non mesurée, pas de ratio entre UUID/instrumentations différents.
4. Réutilisation exacte208 : non qualifiée ; sérialisation identique, première restauration native acceptée ; comparaison complète encore en cours.
5. Coût : points de contrôle208 en 0,121350125 / 0,114084291 / 0,195911333 s ; capture de cache totale 0,061756958 s, première lecture native 0,004909917 s et remise en état complète 0,067238 s ; comparaison du bras complet encore en cours.
6. BénéficeGPU+cache : non mesuré.
7. Deux EOS utiles : non obtenus ; 171 IDs partiels dans la tentative fonctionnelle.
8. Isolation : rejeu natif gelé exact ; historiques complets et isolation après restore non qualifiés.
9. Gardes : ressources/secteur/intégrité validés pour les runs clos ; couvertureUID tardive explicitement limitée.
10. ConfigurationRONIN : CPU10/GPU0, slot1 conservé ; aucune nouvelle sélection production.
11. Pilote fonctionnel : pas prêt sans les deux EOS utiles et le reste de la qualification.
12. Limites600 : modèle110 Go/16 Gio, 20–22min de préfill CPU8, décodage de progression≈0,117 token/s, un slot, GPU/cache complets non qualifiés. Aucune performance600 n’est extrapolée.

## Preuves et intégration

- [Tentative fonctionnelle interrompue](evidence/agents-cpu8-timeout/summary.json).
- [Témoin natif clos](evidence/native-full8-closed/README.md).
- [Composants Q8/Q2 mesurés](evidence/metal-representative-closed/README.md).
- [Reconstruction des sourcesV4](evidence/source-reconstruction/README.md).
- [LanceurUID](UID_SUPERVISOR.md), [audit snapshot](snapshot_audit.md), [audit Metal](metal_audit.md).

Les sources et benchmarks restent expérimentaux. Aucun merge dans main, aucune activation automatique en production, aucun poids ou secret publié. Les métriques non mesurées restent null et les arrêts partiels/error/cancelled conservent leur classification.
