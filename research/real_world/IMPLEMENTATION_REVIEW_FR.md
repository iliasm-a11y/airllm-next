# Revue de l'implémentation de l'évaluation réelle

Revue initiale du 23 octobre 2026, avec compléments liés à leurs preuves.
Périmètre : orchestration des conversations,
campagne supervisée, vérificateur de code et tests associés. Aucune inférence,
aucun chargement de poids, aucun appel MLX/Torch n'a été lancé pour cette revue.
Les reproductions emploient exclusivement des fichiers synthétiques temporaires.
Cette revue d'implémentation ne remplace pas la lecture des réponses du modèle.

**Conclusion : aucun défaut bloquant d'EOS, de conservation du contexte ou de
cap de génération identifié. Deux défauts du vérificateur ont été reproduits et
corrigés. Un risque initial de persistance a été identifié, puis corrigé en
version 2 ; aucune perte de preuve n'a été observée.** Le moteur et le programme
du processus initial n'ont pas été modifiés par cette revue. L'incident ultérieur
du routeur D2 et sa correction sont traités séparément dans
[ROUTER_TIE_FIX_FR.md](ROUTER_TIE_FIX_FR.md).

## Constats et corrections

| Priorité | Constat vérifiable | État |
|---|---|---|
| P2 | `check_real_world_code.py --output` pouvait désigner son fichier `--responses` ou `--tasks` et l'écraser après une exécution réussie. | Corrigé avant emploi du vérificateur sur la campagne : refus des collisions directes, liens symboliques et liens physiques avant toute probe. |
| P2 | Un tour `failed` ou `started` sans champ `text` provoquait un `KeyError` ; aucun rapport de contrôle n'était écrit. | Corrigé : conserver un résultat `executed=false`, `passed=false`, état source, fragment exact, empreinte du fragment et erreur initiale. Aucun fragment n'est exécuté. |
| P2, version initiale | L'ancien `evaluate_real_world.py:24–25` remplaçait directement le JSON cumulatif à chaque token. Une interruption pendant l'ouverture/troncature/écriture pouvait laisser ce JSON invalide et supprimer les tours précédemment présents dans ce fichier. | Aucune perte observée. Corrigé en version 2 après archivage des sources initiales ; la reprise emploie la sauvegarde atomique décrite ci-dessous. |

Les reproductions avant/après, codes de sortie et empreintes des sources sont
dans [implementation_review_probes.json](implementation_review_probes.json).
Avant correction, une sortie égale à l'entrée produisait `exit=0` et remplaçait
le JSON de réponses par un rapport de contrôle. Après correction : `exit=1`,
erreur explicite et contenu original conservé. Un tour synthétique interrompu
produit désormais un rapport valide, avec zéro code exécuté et zéro réussite.

La fonction de contrôle ajoute également
`same_object_as_first_argument` à chaque cas exécuté. Cela permet de relever une
fonction B1 qui renverrait la liste d'entrée elle-même. Les fixtures gelées ne
sont pas modifiées : ce champ est une preuve supplémentaire pour l'appréciation
de l'exigence « nouvelle liste », distincte des seuls tests de valeurs.

## Vérifications de l'orchestration

- Le prompt effectif provient du vrai template, avec système, historique et
  préfixe assistant ; les IDs complets et les paramètres effectifs sont sauvés.
- Le cap demandé et le prompt complet doivent tenir dans le contexte avant
  toute mutation d'état. Un EOS peut arrêter avant le cap ; aucun quota ne force
  la poursuite. Les deux EOS déclarés par le modèle sont traités comme arrêts.
- La session conserve le contenu généré, et le décodage brut inclut le token
  d'arrêt. Le dernier token échantillonné n'a pas encore été exécuté : le tour
  suivant le réintroduit dans son entrée lorsque le préfixe rendu le permet.
- Le cache n'est réemployé que si sa position et tous les IDs consommés
  correspondent exactement au nouveau préfixe. Sinon, la session reconstruit
  son état. L'échec de génération ou du callback ne valide pas un historique
  partiel et force une reconstruction au nouvel essai.
- Le décodage glouton est explicite. Les options de génération non implémentées
  sont refusées ; les paramètres officiels de sampling ne sont pas annoncés
  comme exécutés. Les logits non finis sont refusés avant émission d'un token.

Ces propriétés sont contrôlées par les tests d'orchestration avec modèle factice.
Elles ne constituent pas à elles seules une preuve de concordance numérique du
modèle réel ; l'ablation mesurée doit comparer séparément ses entrées et sorties
effectives.

## Sûreté et périmètre du vérificateur de code

Le code exact produit est conservé avec SHA-256. Le retrait d'une seule enceinte
Markdown est identifié comme extraction et non comme respect de la consigne
« code seul ». Aucun correctif du programme généré n'est effectué.

Les probes réelles vérifient le refus de connexion réseau, d'écriture hors du
répertoire temporaire et de création de processus, ainsi que l'autorisation
d'une écriture dans ce répertoire contrôlé. Python démarre avec `-I -S` ; une
liste AST et une liste explicite de fonctions accessibles limitent le programme
aux petits exercices sans import. Les sorties du processus vont dans des
fichiers soumis à la limite de taille, évitant une capture illimitée en mémoire.

Darwin a refusé les limites AS/DATA demandées : elles sont marquées non appliquées
dans les preuves. CPU 3 secondes, fichiers 1 Mio et 16 descripteurs restent
appliqués. Le parent arrête sur RSS échantillonné au-delà de 256 Mio et sur délai
mural. Un échantillonnage de 10 ms n'est pas une limite d'allocation stricte et
peut manquer un pic bref. Ce dispositif ne doit pas être présenté comme un
environnement général pour exécuter des programmes arbitraires.

## Limites d'évaluation à préserver dans le rapport

- Un rejet AST n'est pas une preuve de code incorrect : la liste interdit
  notamment `type` et certains identifiants commençant par `_`, qui peuvent
  apparaître dans des solutions Python légitimes. Distinguer le refus du banc
  d'essai d'un échec fonctionnel effectivement exécuté.
- L'identité d'une nouvelle liste n'était pas une assertion des fixtures B1.
  Le champ ajouté permet sa vérification sans modifier les demandes gelées.
- B2 contrôle aussi la non-mutation de l'entrée, exigence plus stricte que le
  texte explicite de sa demande. Si ce seul contrôle échouait, il faudrait le
  signaler séparément de l'exactitude de la somme.
- Le délai déclaré de 2 secondes couvre le processus de test et ses quatre cas,
  avec son démarrage. La mention « par cas » du protocole doit être rectifiée
  pour décrire ce périmètre réel. Le plafond CPU distinct est de 3 secondes.
- Les compteurs `evaluated`, `executed` et `passed` sont distincts : un résultat
  enregistré mais non exécuté ne compte pas comme réussite. Une réussite des
  cas déclarés ne donne pas automatiquement une note de respect de toutes les
  instructions ou de qualité rédactionnelle.
- La TTFT moteur exclut la préparation avant sa première passe. Le champ brut
  `perceived_ttft_seconds` est une estimation basse : l'analyse doit conserver
  les bornes et définitions, sans l'annoncer comme chronométrage exact d'une
  interface utilisateur. Le premier processus n'implique pas un cache disque
  froid ; les lectures physiques disponibles concernent le processus, pas une
  mesure certifiée des octets NAND.
- Les maxima RSS/swap sont échantillonnés. Le swap est système : ne pas sommer
  ses maxima par tour et ne pas attribuer toute variation au modèle.

Le comptage du fichier de tâches gelé a été revérifié : **21 sessions, 25 tours,
dont 21 tours principaux ; plafond total 7 296 tokens** avec les quatre paliers
longs. Ces chiffres décrivent les demandes, pas le nombre d'expériences déjà
achevées ni le nombre de réponses réussies.

## Validation exécutée

```text
PYTHONPATH=air_llm python -m pytest --noconftest \
  tests_next/test_gguf_chat.py tests_next/test_real_world_code_checks.py -q
42 passed in 1.51s
```

L'exclusion du conftest évite les imports des backends de modèle pour cette
vérification légère. Les tests couvrent notamment EOS, dernier token non
consommé, historique réel, divergence du préfixe, cap de contexte, reprise après
échec, callbacks en erreur, logits non finis, refus AST, code volontairement
incorrect, non-mutation, identité de liste, interruption d'une boucle et
collisions de fichiers.

## Sauvegarde atomique adoptée après archivage du programme initial

Pour remplacer `save` dans `evaluate_real_world.py`, publier un nouveau fichier
dans le même répertoire puis effectuer un remplacement atomique. L'ancien JSON
reste alors valide si le processus est interrompu pendant l'écriture du nouveau.
Le correctif n'a pas été injecté dans le processus initial. Il a ensuite été
adopté pour les processus suivants, après archivage de cette version initiale.
La reprise D2 déclare le hash `5b9d6f72a7ecb9c4e62917defcf99ae2b9aeade0ef197d831850cf4d4cba3e17`,
concordant avec les sources de la version atomique.

```python
from tempfile import NamedTemporaryFile

def save(path, obj):
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                prefix=f'.{path.name}.', suffix='.tmp',
                                delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
```

Ce correctif protège contre l'interruption du processus pendant l'écriture ;
il ne prétend pas certifier la durabilité après coupure d'alimentation. Il ne
remplace pas la conservation distincte des exécutions interrompues et des relances.

Le patch de préparation est conservé localement sous `atomic_json_save.patch`
et n'est pas nécessaire au dépôt public une fois le correctif adopté.
Les [9 tests dédiés](tests/test_atomic_json_save.py) ont passé sur des copies
temporaires du programme, y compris erreur d'écriture partielle, sérialisation
invalide, fermeture en erreur et refus de remplacement : l'ancien JSON reste
intact et le temporaire est nettoyé. Les NaN/Inf ne peuvent pas être publiés comme
JSON standard. Les empreintes avant/après attendues et la preuve que le programme
gelé est resté inchangé figurent dans
[atomic_json_save_manifest.json](atomic_json_save_manifest.json), qui est le
manifeste historique de préparation avant adoption. Le programme initial archivé
conserve la preuve de la différence.

## Empreintes à la clôture de la revue initiale

| Fichier | SHA-256 |
|---|---|
| `gguf_chat.py` | `6af7f468b1de6282eb366fd8918732bd5f95792e4a5bad1e5b2b8a11b5e15074` |
| `evaluate_real_world.py` | `8aefd03a9e15ba05d9fe14146d0f5b74ec5f1aad2c81a77f9b306e9cb4d11f27` |
| `check_real_world_code.py` | `b7fd48405a7946bd00aae09f3f9e2c2bd4c73240ef896fda3865bfafaf278c63` |

Les empreintes complètes des deux fichiers de tests sont également enregistrées
dans le JSON de reproduction. Une modification ultérieure impose de distinguer
la nouvelle version de celle relue ici.

## Contrôles supplémentaires sur les codes effectivement générés

Les programmes B1 et B2 complets ont ensuite été lus et exécutés sans correction
dans le même bac à sable. La preuve conserve leur source exacte et son SHA-256,
les représentations typées des arguments/résultats, les assertions, les sorties
du processus et les probes de sûreté :
[`adversarial-code-checks.json`](../../evidence-real-world/adversarial-code-checks.json).

- B1 : **56/56 cas** du domaine annoncé — entiers, chaînes, listes, dont listes
  imbriquées et mélanges déterministes — respectent valeurs et ordre, ne
  modifient pas l'entrée et retournent chaque fois une nouvelle liste.
- B2 : **35/35 cas d'entiers**, dont négatifs, zéros, grands entiers et séries
  déterministes, donnent les sommes attendues sans mutation de l'entrée.

Une hypothèse de collision entre une liste et un tuple a aussi été vérifiée :
`[[1], (1,)]` n'est pas traité comme deux valeurs distinctes par B1. Les tuples
ne font cependant pas partie des types annoncés dans la demande initiale.
Les trois cas de cette extension sont donc enregistrés séparément, **sans être
comptés comme échecs de la tâche B1**. Aucun correctif de la réponse ou du modèle
n'est justifié par ce test hors périmètre.

Ces cas sont des contrôles complémentaires, pas de nouvelles tâches confiées
au modèle, ni une démonstration exhaustive de correction. Reproduction :
`research/real_world/check_adversarial_code.py` ; aucun prompt ni fichier de
tâches gelé n'a été modifié.
