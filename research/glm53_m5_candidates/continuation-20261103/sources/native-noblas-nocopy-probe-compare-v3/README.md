# Comparateur de fenêtre NoCopy 128 — source v3

Préparé, non importé/non testé/non exécuté. Aucun AST, build, appel natif,
lecture de payloads ou GPU pendant C16. Les anciens comparateurs, sources et
preuves NoCopy restent inchangés. Ce dossier ne contient que la révision du
comparateur et son validateur du nouveau garde de build ; le runner et le
helper sont ceux du candidat NoCopy précédemment relu, consommés en lecture.

Deux corrections de raccordement : le helper est identifié par son rôle
explicite hashé, et un driver favorable n'est accepté que si le vrai garde
`native-nocopy-build-supervisor-v2` a clos toutes ses phases et son postflight.
Un fichier `guard_pending`, un postflight refusé, des descendants forcés ou
restants, une observation FD manquante, des logs non appariés ou un SHA différent
font refuser la liaison. Le validateur rejoue la politique mémoire native
originale aux timestamps réellement enregistrés et conserve les plafonds
8,5 GiB/32 MiB ainsi que l'enveloppe de build plus stricte de 2 GiB.

Le reçu du garde et les empreintes de sa source/manifest sont liés dans une
enveloppe de provenance séparée. La carte des sources mathématiques reste
exactement celle produite par le runner : pas de faux ajout à sa clé. Les deux
builds, leurs propres clés, profils, gardes et 118 séries brutes restent requis.
Cette révision ne lance jamais un build, un modèle ou une requête HTTP.

Une fenêtre identique signifie seulement que les tokens 0..127 de la requête0
ont passé ce contrôle borné. Les 600 entrées gelées restent intégrales ; aucune
continuation, logits finaux, fin de prompt, EOS utile ou qualification complète
n'est produite. CPU bit à bit et tolérances Metal 1e-3/1e-4 sont rapportés
séparément, sans remplacer le critère strict de rejet de cette fenêtre.

Commande future de liaison, uniquement après vrai build revu/gardé et sans
aucune inférence :

```text
/Users/macbook/Documents/Codex/2026-11-02/files-pasted-by-the-user-mission/work/venv/bin/python /Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/native-noblas-nocopy-probe-compare-v3/probe_compare.py --make-binding --output /Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/native-noblas-nocopy-probe-compare-v3/actual-build-binding.json
```

Les valeurs par défaut désignent le nouveau build supervisé `build-run1` et
son profil, pas le package clos original. La comparaison future exige ensuite
`--reference`, `--candidate`, les deux `--*-guard`, `--binding` et `--output`.
Aucune liaison réelle, aucun build ni résultat numérique nouveau n'existe à ce
stade. La revue indépendante de ces deltas reste à obtenir avant lancement.
