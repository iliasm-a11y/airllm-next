# Audit de publication — instantané pendant les essais

Cet audit porte sur les fichiers ajoutés ou modifiés pour cette mission. Il ne
déclare pas la campagne terminée et ne remplace pas son rapport final. Aucune
nouvelle inférence n'a été lancée pour l'audit. Les fichiers du moteur et du
programme de campagne gelés n'ont pas été modifiés.

## Confidentialité et contenu des fichiers

Le premier scan ciblé a couvert 90 fichiers ajoutés ou modifiés, en recherchant
des formes de clés API, tokens GitHub/AWS, clés privées, identifiants intégrés
dans des URL, affectations littérales de secrets, adresses email et téléphones
français. **Aucun candidat de ces catégories n'a été repéré.** C'est un résultat
de scan ciblé, pas une garantie générale sur tout l'historique du dépôt.

Les prompts sont synthétiques ; les prénoms et organisations présents dans les
cas d'essai sont fictifs. Aucun contenu de messagerie, calendrier ou document
personnel n'a été utilisé. Les métadonnées examinées ne contiennent pas de
numéro de série matériel ni d'adresse réseau personnelle.

61 occurrences de chemins absolus `/Users/macbook/...` figuraient dans ce
premier instantané : commandes, chemins de sources, modèle local et traces
d'erreur. Ce sont des informations de reproduction, pas des identifiants de
connexion. Les preuves brutes n'ont pas été anonymisées a posteriori, pour ne pas
changer silencieusement leurs empreintes. Leurs chemins ne seront évidemment
pas utilisables tels quels sur un autre ordinateur.

Les binaires et archives déjà suivis dans les anciennes branches — petits
fixtures de test, roues, anciens résultats — ne sont pas de nouveaux ajouts de
cette mission. Cet audit ne propose pas de supprimer ce contexte hérité.

## Ce qui reste local

- **GGUF complet et tokenizer de référence** : aucun nouveau checkpoint n'est
  ajouté à la PR. Le modèle est référencé par identité, taille et empreinte.
- **`evidence-real-world/*.npz` et `*.npy`** : non publiés selon `.gitignore`.
  En particulier, le NPZ du routeur contient une véritable matrice de poids de
  gate de 2 Mio, ainsi que des états numériques. Son fichier compressé mesure
  1 004 681 octets. Le JSON public conserve schéma, dimensions et SHA-256 ; le
  programme de replay public permet de refaire la capture avec le modèle local.
- **`evidence-real-world/*.xml`** : détails locaux des tests. Les journaux et
  gardes JSON publics gardent résultats, limites et états de sortie.
- **`research/real_world/atomic_json_save.patch`** : préparation historique,
  volontairement non staged. Le correctif est déjà adopté et les sources v1
  sont archivées ; le manifeste conserve les empreintes avant/après. Le patch
  n'a pas été supprimé. Les tests publics exercent l'implémentation v2 actuelle,
  sans avoir besoin de ce patch historique.
- Répertoires de travail, environnements, caches Python/pytest et fichiers
  temporaires : exclus. Les temporaires atomiques de la campagne sont ignorés.

Les captures NPZ et XML sont désormais décrites comme fichiers locaux dans les
docs d'audit ; les liens publics pointent vers les JSON et journaux disponibles.
Aucun poids n'a été publié pour corriger un lien.

## Test public du routeur et preuves locales

La régression de la vraie capture doit recevoir explicitement
`AIRLLM_TEST_ROUTER_CAPTURE`. Sans cette variable, **seul ce test est ignoré** ;
les tests synthétiques CPU continuent de s'exécuter. Un chemin explicitement
fourni doit exister, respecter la limite de 8 Mio et correspondre à l'empreinte
attendue. Aucun fichier de taille arbitraire n'est chargé comme fixture.

La suite locale avec capture a passé 13 tests avant cette modification limitée
à la découverte du fichier de test. La version publique ciblée donne ensuite
12 tests réussis et 1 ignoré, avec le motif explicite. Aucun changement du moteur
ou de ses paramètres n'est intervenu entre ces vérifications. Les preuves sont
`router-fix-tests-batch.log/.guard.json` et
`router-public-tests.log/.guard.json`.

La vérification publique CPU a démarré juste avant réception de la consigne de
différer ces tests pendant l'inférence en cours ; elle a duré 1,14 s au total,
sans GPU, RSS échantillonné 238 Mo, réserve disponible minimale 3,98 Go et
croissance de swap nulle. Cette courte concurrence est déclarée ; elle ne doit
pas être présentée comme une expérience de performance isolée. Aucun autre test
Torch/MLX ne sera lancé par cet audit pendant les inférences.

Commande publique ciblée à employer hors inférence, sans activer de capture :

```sh
PYTHONPATH=air_llm python -m pytest --noconftest \
  tests_next/test_qwen_router_ties.py -q -rs
```

Le résultat global antérieur de validation conserve sa valeur pour sa version
enregistrée. Ne pas lui attribuer rétroactivement cette modification minimale
du test ; les journaux distinguent les deux révisions.

## Cohérence documentaire

- La revue d'implémentation est maintenant explicitement historique : risque
  d'écriture v1 identifié, puis sauvegarde atomique adoptée v2 après archivage.
  Son tableau de hashes initiaux n'est pas présenté comme l'état courant.
- Le diagnostic du routeur décrit le repli **gate CPU complète**, pas seulement
  le départage de probabilités natives. Les temps globaux avant/après ne sont
  pas traités comme un surcoût causal du correctif.
- La réponse D2 est conservée. « Date actuelle : 8 avril » diffère du critère
  prévu « événement au 22 mai », mais la formulation initiale est ambiguë ;
  cela ne prouve pas à soi seul une erreur de lecture du modèle. Un essai avec
  demande clarifiée doit rester une observation distincte.
- Les cas liste/tuple de B1 sont une extension hors contrat. Ils ne deviennent
  pas des échecs de sa consigne initiale entiers/chaînes/listes.
- Les maxima mémoire, le swap système, les lectures du processus et les temps
  d'initialisation gardent leur périmètre de mesure. Aucune validation de SSD
  externe, d'iPhone, d'un autre modèle ou de trafic NAND n'est revendiquée.

Les documents `ANALYSIS_FR.md` et `RESPONSES_FR.md` sont des instantanés générés
pendant les essais : ils doivent être régénérés après les dernières réponses,
annotations et gardes, pas promus tels quels comme rapport définitif.
Le générateur d'analyse contient encore une description courte du repli comme
« départage de probabilités égales » ; le responsable de ce générateur doit
l'aligner sur le contrat de gate CPU complète avant la dernière génération.

Le rapport demandé `REAL_WORLD_EVALUATION_FR.md` reste à produire ou à constater
à son emplacement final. Les figures sont optionnelles et annoncées absentes
tant qu'aucun export réel n'existe : ne publier ni courbe factice ni résultat de
rendu non exécuté.

## Fichiers à inclure et contrôle final

La liste exacte des ajouts/modifications utiles **observés à cet instantané** est
dans [PUBLICATION_FILES.txt](PUBLICATION_FILES.txt), avec catégories et exclusions
dans [publication_manifest.json](publication_manifest.json). Elle conserve les
échecs, pilotes, refus d'admission, sources archivées et preuves de correction ;
les copies de campagne sont des archives et ne comptent pas comme observations
supplémentaires.

Cette liste n'est pas un `git add` automatique et ne prétend pas inclure des
résultats encore non produits. Avant la PR, la rafraîchir avec les derniers
fichiers du rapport, les gardes terminés, les diagnostics réellement exécutés et
les annotations finales. Vérifier ensuite le diff staged, les références et les
empreintes. Ne pas ajouter en bloc tous les fichiers locaux.

Les espaces de fin de ligne du transcript brut sont intentionnels : ils
proviennent du texte du modèle, notamment de ses sauts de ligne Markdown. Le
premier scan en compte huit lignes dans `RESPONSES_FR.md` ; ce nombre peut évoluer
avec les réponses. Il faut conserver cette preuve et documenter les lignes
signalées par `git diff --check`, sans masquer globalement les contrôles ni
réécrire silencieusement les réponses. Le patch historique non staged n'entre
pas dans ce diff.

Aucun fichier n'a été retiré par cet audit. Aucune campagne en cours n'est
qualifiée d'achevée sur la seule base de cette liste de publication.

<!-- BEGIN FINAL PUBLICATION REFRESH -->

## Annexe de clôture — inventaire de publication rafraîchi

Instantané UTC : 2026-10-23T21:14:57.699644+00:00. Le corps précédent conserve son statut historique.
Les indications antérieures « en cours », « à produire » et les anciens nombres de fichiers
décrivent cet ancien instantané. Le rapport final et cet inventaire daté décrivent le nouvel état constaté.

Rapport final constaté : ../../REAL_WORLD_EVALUATION_FR.md ; SHA-256 : 1c165b1a95417e300ced9213edd08aa723271ccbe7b2b17e61157f10056fc975.
Périmètre : fichiers ajoutés/modifiés non ignorés de cette mission relativement à HEAD,
avec les trois documents de publication régénérés. Ce n’est ni un ajout Git automatique
ni une publication. Aucun modèle, test ou nouvelle inférence exécuté par ce rafraîchissement.

Proposition : 146 fichiers. Scan ciblé : 144 textes ; 0 candidats de secrets/données personnelles. Seuls fichier, ligne et catégorie sont enregistrés.
L’absence de candidat ne garantit pas l’absence de tout secret ou donnée personnelle.
Les chemins absolus techniques restent dans les preuves brutes ; aucune anonymisation silencieuse.

Campagne principale : 21/21 observations achevées uniques. Annotations relues depuis independent_review.json : echec_critere=2, indetermine=1, partiel=2, reussi=16.
Ce comptage n’invente aucune note et ne mélange ni pilote, ni ablation, ni contrôle clarifié.
A1 garde son adjudication ; l’ambiguïté initiale de D2 reste distincte du contrôle clarifié.

| Exécution | Statut effectif | Tours achevés / présents | Garde | Swap maximal octets |
|---|---|---:|---|---:|
| campaign-before-router-fix.json | failed_guard | 8/9 | 1 | 0 |
| campaign-resumed.json | completed | 13/13 | 0 | 0 |
| campaign.json | failed_guard | 8/9 | 1 | 0 |
| concision-control.json | completed | 1/1 | 0 | 0 |
| conversation-rebuild.json | completed | 5/5 | 0 | 0 |
| d2-clarified.json | completed | 1/1 | 0 | 0 |
| f2-concision-control.json | completed | 1/1 | 0 | 0 |
| hard-reasoning.json | completed | 1/1 | 0 | 0 |
| long-1024.json | interrupted_guard | 0/1 | system_swap_growth | 170459136 |
| long-128.json | completed | 1/1 | 0 | 0 |
| long-256.json | completed | 1/1 | 0 | 0 |
| long-512.json | completed | 1/1 | 0 | 0 |
| pilot-short-caps.json | failed_guard | 2/3 | -15 | 0 |
| preflight-source-change.json | failed_guard | 0/1 | -15 | 0 |

Le palier 1 024 reste NON VALIDÉ : tokens observés [244], garde system_swap_growth, swap maximal 170459136 octets. Aucune nouvelle tentative n’est lancée ou recommandée par cet outil.

Ablation E : 5/5 comparaisons achevées strictement identiques en IDs et texte ; détail dans le manifeste.
Les contrôles A2/F2 de concision, D2 clarifié et A4 gardent leur propre exécution et leur
annotation. Leur présence ou leur réussite n’est pas déduite du seul nom du fichier.

Exclus : poids/checkpoints, captures NPZ/NPY, XML locaux, caches et work/, fichiers
temporaires, patch atomique historique déjà appliqué. Aucun fichier local n’a été supprimé.
Le test réel du routeur reste opt-in via AIRLLM_TEST_ROUTER_CAPTURE ; sans capture,
il est ignoré explicitement et les tests synthétiques restent disponibles.
Les SHA des journaux/tests prouvent leur version constatée, sans relancer leur validation.
Les temps globaux avant/après correction ne sont pas un surcoût causal du repli routeur.

Espaces finaux du transcript brut conservés : 17 lignes. Le manifeste indique les numéros ; aucune règle globale ne masque le contrôle.
Le manifeste est écrit en dernier. Il ne contient pas sa propre taille/empreinte,
pour éviter une dépendance circulaire ; la liste et cet audit y sont normalement hachés.

Conclusion de périmètre : inventaire et contrôles locaux terminés sur les fichiers constatés ; revue du diff staged reste distincte.
Les détails techniques, exclusions et constats sans valeurs secrètes sont dans publication_manifest.json.
Ces contrôles ne valident ni autre modèle/matériel, ni SSD externe, ni activité physique NAND,
ni sûreté générale de code arbitraire ou absence absolue de données personnelles.

<!-- END FINAL PUBLICATION REFRESH -->
