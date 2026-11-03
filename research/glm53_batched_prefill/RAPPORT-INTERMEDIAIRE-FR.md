# RONIN M5 — résultats clos et validation en cours

État du 3 novembre 2026. Les objectifs A (GPU complet), B (cache exact utile) et C (deux agents utiles jusqu’à EOS) restent indépendants et non qualifiés à cette étape. Les composants synthétiques, les captures de fidélité et les tâches fonctionnelles ne sont pas interchangeables.

## Résultats matériels clos

| Essai | Résultat et portée |
|---|---|
| RONIN fonctionnel CPU10/GPU0, batch8, Max | Agent  1 : 522 tokens, préfill 1238, 15 911 s, 171 IDs puis délai 2700 s ; aucun EOS ni réponse finale utile. Agent  600 et retour1 non lancés. |
| Témoin natif CPU8 sans cache | Trois prompts gelés1 → 600 → 1 : 521/542/521 tokens, préfills1285, 763 678/1322, 539 045/1314, 183 277 s ; quatre vrais échantillons greedy diagnostiques par requête. Code0, secteur/intégrité validés ; aucune qualification fonctionnelle EOS. |
| Rejeu numérique du témoin | Agent  1 request0 contre2 : 154 880 logits finis, état sérialisé, captures, 15 642 écritures observées, continuations et post-état exacts. Rejeu intra-bras, pas comparaison indépendante de variantes. |
| Préfixe commun |600 identités natives Max, 211 IDs communs. Trois sérialisations complètes à208 identiques : 156 739 588 octets(149, 478 519 Mio). Aucun restore réel encore exécuté. |
| Q8 Metal représentatif | M1536/K4096, N8/16/32/64/128 : 380 928 sorties bit à bit face à ARM nrc2 et au graphe CPU10. Poids/activations synthétiques, pas un préfill GLM complet. |
| Q2_K Metal représentatif | M2048/K4096, N8/16/32 : 114 688 sorties bit à bit face au dot ARM nrc1. Poids synthétiques, quantification d’activation CPU native. Pas de routes/logits/états de modèle mesurés. |

Les tolérances Metal existantes sont aussi satisfaites sur ces composants. Elles ne sont pas élargies. Les premiers petits tests Q8 M10/14/32 avaient montré des différences avec certains chemins du dispatcherCPU : celles-ci ne constituent pas à elles seules un rejet global GPU. Aucune mesure de composant n’établit un gain CPU/GPU sur les prompts complets. Les poids préparés sont chauds ; aucune accélération du SSD n’en est déduite.

## Ressources et provenance

Les deux campagnes GLM closes respectent les gardes de ressources observées : RSS échantillonnée maximale6029656064 octets pour la tentative fonctionnelle, 5 608 177 664 pour le témoin natif ; croissance de swap observée0 dans les deux cas. Les alertes de pression 2 sont conservées et admises seulement dans le domaine contextuel du watchdog existant. Cela ne signifie pas que la pression est restée normale. Les limites 8, 5 Gio/32 Mio, les réserves, contrôles secteur/couvercle/horloges, intégrité et identitéPID ne sont pas désactivés.

Le verrou UID du premier témoin natif a été acquis tardivement et n’a pas été transmis rétroactivement au helper. Cette limite est conservée dans son reçu. Le nouveau superviseur acquiert la propriété avant lancement et transmet son descripteur ; ses tests synthétiques couvrent également split. L’essai CPU16 en cours a en plus une observation OS du descripteur réel dans le helper.

Le modèle original Q2_K_S reste externe et inchangé. Le runtimeV4 hérité est épinglé à llama.cpp 3f8af15a76ac8675812e856da45454bafc90f12c avec16 patches. Une reconstruction de sources isolée a retrouvé l’ordre et les16 SHA des patches, puis les13 empreintes finales du build. Aucun runtime natif n’a été reconstruit pour cette vérification. Les bibliothèques effectivement mesurées et les en-têtes du helper restent identifiés séparément ; une copie de sources ne constitue pas une validation d’un nouveau binaire.

## CPU16 et cache

L’alimentation secteur a été rétablie après le refus sur batterie. CPU16 est désormais en cours avec le même helper, wrapper, runtime et prompts gelés que CPU8. Le contrôle précoce des32 puis128 premiers tokens passe sur118 noms de routes/routeurs/KDA avec le critèreCPU bit à bit ; les états complets à128 et208 correspondent au témoin. Ce résultat ne qualifie ni le prompt entier, ni les trois requêtes, ni un gain.

Après clôture, seules une comparaison complète et des gardes valides permettront de retenirCPU16. La restauration devra ensuite être testée avec son propre témoin de mêmeprofil et les mêmes dépendances. Le snapshot reste un seul objet immuable de256 Mio maximum ; scratch, états actifs, graphes et captures sont comptés séparément. La séquence native ne sauvegarde ni sampler ni logits : suffixe privé non vide et sampler neuf restent obligatoires. PARTIAL_ONLY n’est pas une base autonome après reset.

Pour32+, les frontières du helper split à208 diffèrent de full et certains checkpoints requis ne sont jamais atteints. [L’audit statique](split32_protocol_audit.md) décrit ce blocage structurel. L’admission du mode split par le lanceur ne corrige pas le protocole numérique. Aucun essai long de ce chemin ne doit être lancé comme s’il était qualifiable en l’état ; CPU16→restore16 reste aligné et inchangé.

## Verdicts aux douze questions

1. GPU plus rapide sur GLM complet : non démontré ; composants multi-token exacts seulement.
2. Meilleur batch : inconnu ; CPU8 témoin clos, CPU16 en cours.
3. Économie de préfill : non mesurée, pas de ratio entre UUID/instrumentations différents.
4. Réutilisation exacte208 : non qualifiée ; sérialisation identique, zéro restore réel.
5. Coût : points de contrôle208 en0, 121 350 125/0, 114 084 291/0, 195 911 333 s ; création de cache et restauration encore non mesurées.
6. BénéficeGPU+cache : non mesuré.
7. Deux EOS utiles : non obtenus ; 171 IDs partiels dans la tentative fonctionnelle.
8. Isolation : rejeu natif gelé exact ; historiques complets et isolation après restore non qualifiés.
9. Gardes : ressources/secteur/intégrité validés pour les runs clos ; couvertureUID tardive explicitement limitée.
10. ConfigurationRONIN : CPU10/GPU0, slot1 conservé ; aucune nouvelle sélection production.
11. Pilote fonctionnel : pas prêt sans les deux EOS utiles et le reste de la qualification.
12. Limites600 : modèle110 Go/16 Gio, 20–22min de préfill CPU8, décodage de progression≈0, 117token/s, un slot, GPU/cache complets non qualifiés. Aucune performance600 n’est extrapolée.

## Preuves et intégration

- [Tentative fonctionnelle interrompue](evidence/agents-cpu8-timeout/summary.json).
- [Témoin natif clos](evidence/native-full8-closed/README.md).
- [Composants Q8/Q2 mesurés](evidence/metal-representative-closed/README.md).
- [Reconstruction des sourcesV4](evidence/source-reconstruction/README.md).
- [LanceurUID](UID_SUPERVISOR.md), [audit snapshot](snapshot_audit.md), [audit Metal](metal_audit.md).

Les sources et benchmarks restent expérimentaux. Aucun merge dans main, aucune activation automatique en production, aucun poids ou secret publié. Les métriques non mesurées restent null et les arrêts partiels/error/cancelled conservent leur classification.
