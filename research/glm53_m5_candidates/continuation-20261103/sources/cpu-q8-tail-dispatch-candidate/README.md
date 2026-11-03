# Diagnostic Q8 CPU : lots intérieurs et queues

Source seulement, non construit/non exécuté. AST Python vérifié ; aucun appel
natif, build ou matériel n'a été lancé. `prepare-manifest.json` ferme les sources.

42 cas : M8192/K4096, les21 N demandés et graines5305/713962. Un unique backend
CPU10 direct, `use_ref=false`, traits Q8 natifs inchangés et quantification native
des poids ligne par ligne. Aucun scheduler, initialisation BLAS, load_all ou GPU.

Pour chaque cas, les mêmes poids et activations comparent fullN et la politique
proposée v4 (floor16 pour >16 nonmultiple16, puis queue impaire9..15→8+reste)
contre8+reste. Le rapport conserve bitcounts, première différence, maxabs, SHA
des sorties finies et les partitions. Il **n'exporte aucun F32** : les comptages
sont effectués dans le natif, sans possibilité de les recomparer depuis des raws.

`native_vec_dot_nrows` vient réellement de la bibliothèque CPU. Les champs nrc
par chunk sont une dérivation explicitement nommée du source V4, pas une trace
instrumentée du noyau. Aucun trait, kernel ou bibliothèque n'est modifié.

Poids35 651 584B ; entrée maximale2 097 152B ; trois sorties12 582 912B ; deux
vecteurs de comparaison8 388 608B ; metadata2 097 152B et ligne16 384B, soit
60 833 792B de payload/enveloppe avant padding et workspace backend. Le code
vérifie aussi son allocation réelle de tenseurs contre le plafond logique64MiB.
La résidence réelle reste inconnue : plafond getrusage natif et suivi externe
128MiB, réservé en plus des gardes originaux8,5GiB/swap32MiB/AC/headroom. Un
dépassement est conservé comme refus/échec, sans relèvement de plafond.

Le contrôleur reprend le vrai GO/PID/birth/exe/FD UID observé du v2 BLAS, sous
le superviseur et son BENCHMARK_LOCK/verrou UID. Il ne prend pas un second verrou
déjà possédé par le superviseur. Aucun détenteur étranger n'est attendu ni tué.
Durées réelles inconnues ; build plafonné120s, diagnostic120s, sampling0,1s.

Commandes futures, **à faire relire avant exécution** (chemins absolus à utiliser) :

```text
PYTHON work/guarded_build_once.py --output work/cpu-q8-tail-dispatch-candidate/build-guard.json -- PYTHON work/cpu-q8-tail-dispatch-candidate/build_only.py
PYTHON work/guarded_build_once.py --output work/cpu-q8-tail-dispatch-candidate/run1-guard.json -- PYTHON work/cpu-q8-tail-dispatch-candidate/run_only.py --guard-receipt work/cpu-q8-tail-dispatch-candidate/run1-guard.json
```

PYTHON est `/Users/macbook/Documents/Codex/2026-11-02/files-pasted-by-the-user-mission/work/venv/bin/python`.
Les chemins `work/` sont relatifs à la racine de cette mission ; le contrôleur
contrôle les chemins absolus effectifs. La commande CLI exacte du superviseur
est à conserver telle qu'exposée par sa source avant lancement.

Une différence fullN est un résultat diagnostic attendu possible. Une différence
v4 fait refuser la barrière de forme. Même42 cas v4 exacts ne qualifient ni le
modèle, ni ses autres formes/états, ni le protocole HTTP ou la performance.
