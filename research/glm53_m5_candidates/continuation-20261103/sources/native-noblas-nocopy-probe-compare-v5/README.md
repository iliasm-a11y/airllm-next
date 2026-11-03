# Comparateur NoCopy 128 — alias KDA exacts v5

Cette révision conserve les sources et reçus INVALID v3/v4. Le vrai producteur
émet 152 noms par chunk : 118 séries canoniques et 34 alias KDA dont le nom se
termine exactement par ` (reshaped)`. Le cas clos run3 contient quatre chunks
de 32 tokens, soit 608 lignes et 136 alias. Aucun résultat numérique ne découle
de ce constat de métadonnées.

Origine vérifiée dans les sources disponibles du build NoCopy et identiques à
la reconstruction V4 : `src/models/glm5-next.cpp:884` nomme la sortie KDA,
`:891` la passe à `ggml_reshape_3d`, et `ggml/src/ggml.c:3815` ajoute ce suffixe.
Le helper réellement lié, `native_prefill.cpp:90`, sélectionne tout nom qui
commence par `kda_scan_out-`, puis écrit un fichier distinct avec ses métadonnées
et son SHA. Les chemins et empreintes sont dans `alias-producer-audit.json`.

Le nouveau gate des headers n'accepte que le catalogue canonique et les 34
noms `kda_scan_out-L (reshaped)` des couches L de 0 à 44 hors L modulo4 = 3.
Chaque nom doit exister exactement une fois à chaque chunk attendu, uniquement
dans la requête0/phase `probe128`/positions0..127. Aucun autre extra n'est ignoré.
Types entiers exacts, schéma, fichiers de dump distincts, SHA hexadécimal,
shape/strides/type/taille KDA contigus et paire alias/canonique sont contrôlés.

Avant la comparaison numérique, le validateur recontrôle les fichiers bruts
réels : tailles, SHA, identité stable et égalité de tous les octets entre alias
et canonique, avec deux buffers de 1 Mio au maximum. Les alias sont ensuite
rapportés séparément : 34 noms et, si les preuves passent, 136 lignes sur quatre
chunks. Ils n'ajoutent pas de séries numériques et ne remplacent aucun des
118 contrôles canoniques, du checkpoint128 ou des gardes existants.

Le lecteur CPU8 historique de v4 reste exact et borné ; `build_guard.py` est
copié sans modification. Les clés mathématiques et critères CPU bit à bit /
Metal 1e-3 et 1e-4 restent inchangés. Une fenêtre128 n'est jamais une preuve de
prompt complet, EOS, performance ou préfill complet hors mémoire.

Root recrée la liaison v5 puis lance une seule comparaison des captures closes
avec les commandes de `selection.json`, après contre-revue. Aucun modèle n'est
à relancer. Les contrôles AST et de headers synthétiquement altérés sont
consignés séparément ; ils ne lisent pas les gros fichiers de tenseurs et ne
prouvent pas encore l'égalité des payloads réels.
