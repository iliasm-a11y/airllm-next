# Audit PR6 — packed IO / budgets / durée de vie

Audit de la source exacte `4ed1a58b460f8b93432e6cabc786c4ce9fc732c7` dans `work/pr6-audit`. Correctifs ciblés dans `work/airllm-next`. Aucun modèle exécuté par cet audit ; un microtest MLX de 8 buffers de 144 octets a été lancé. Aucun plafond augmenté, aucun `gc.collect()` ajouté au runtime, aucune nouvelle dépendance installée, aucun commit effectué par cet agent.

## Résultats vérifiés

- Nouveau fichier `tests_next/test_packed_io_safety.py` exécuté contre PR6 exacte : **21 échecs, 33 succès**. Journal `work/pr6-safety-before.txt`.
- Après correction, tests sécurité + lecteur GGUF + cache adaptive + tests originaux PR6 inchangés : **120 succès, 3 ignorés**, 0,24 s. Journal `work/pr6-safety-after.txt`.
- Les trois ignorés correspondent aux parités optionnelles exigeant le paquet upstream `gguf` et à la lecture d'un checkpoint réel explicitement opt-in. Aucun test ignoré n'est présenté comme réussi.
- Vérification `git diff --check` ciblée sans erreur.
- Python utilisé : `/Users/macbook/Documents/Codex/2026-10-22/files-pasted-by-the-user-mission/work/validation-venv/bin/python` (3.12.13). `air_llm/setup.py` impose déjà Python >= 3.12.

Commande lecteur légère utilisée, sans chargement des conftest Torch/MLX :

```sh
PYTHONPATH=air_llm /Users/macbook/Documents/Codex/2026-10-22/files-pasted-by-the-user-mission/work/validation-venv/bin/python -m pytest --noconftest tests_next/test_packed_io_safety.py tests_next/test_gguf.py tests_next/test_adaptive_cache.py air_llm/tests/test_packed_io_copyfree_regression.py -q
```

## P1 — la weakref de l'ndarray ne suit pas les bytes sous-jacents

Dans PR6, `copy=False` construit `owner=np.frombuffer(raw)` puis `out=owner.reshape(...)`. La finalisation est attachée à `owner` mais `out.base.base` expose les `bytes` de `os.pread`. Garder ces bytes, ou `memoryview(out.base.base)`, puis supprimer `out` libère la charge de 144 octets alors que le payload reste vivant. Le budget autorise alors un second buffer avec un premier toujours détenu. Le test original PR6 ne vérifie que la suppression de l'array et ne couvre pas l'export de sa base.

Correction : provider buffer weakrefable `_PackedBuffer` fondé sur le protocole Python 3.12 / PEP 688. Les arrays NumPy et leurs memoryviews gardent ce provider dans la chaîne de propriété. La finalisation est attachée à ce propriétaire. Le chemin `copy=False` partage les mêmes bytes immuables sans copie supplémentaire ; le chemin par défaut conserve une copie writable via `bytearray`.

Vérifié avec les quatre formes de rétention (`base`, `memoryview(base)`, `memoryview(array)`, `np.frombuffer(array)`), avec/sans budget propriétaire et avec/sans copie. L'échec d'admission survient avant le second read tant qu'une vue reste vivante. Le flag writable ne peut pas être réactivé pour `copy=False`.

## P1 — cause reproduite de l'épuisement réel PR6 : cycle dans retained_size

Le diagnostic réel exécuté par le parent, `work/evidence-5.1/pr6-copyfree-diagnostic.json`, montre :

- cache : 66 781 952 octets ;
- budget packed live : 83 361 792 / capacité 83 886 080 octets ;
- 104 loads copyfree ; 28 rejets de cache adaptive ;
- `packed_host_copy_bytes_avoided = 166 723 584`, soit exactement deux fois tous les payloads encore vivants ;
- échec `GGUFMemoryError: Packed buffer budget exhausted` pendant le préfill ; aucun token émis.

`adaptive_cache.retained_size()` définit une fonction locale récursive `visit`. Sa cellule de fermeture capture `visit` elle-même et la liste `arrays` contenant chaque ndarray inspecté. Une fois la fonction retournée, cela forme un cycle inaccessible qui garde les buffers jusqu'au prochain passage du GC cyclique, même pour les tuiles rejetées par l'admission.

Reproduction isolée exacte, sans MLX et sans modèle : 8 payloads de 144 octets inspectés par `retained_size`, puis suppression de toutes les références utilisateur, GC automatique temporairement désactivé dans le seul microtest : **1 152 octets / 8 propriétaires encore vivants**. Après `gc.collect()` : **0 octet / 0 propriétaire**. Preuve : `work/pr6-adaptive-lifetime.txt`.

Un microtest MLX distinct `mx.array(host)` + `mx.eval(...)` sur 8 buffers de 144 octets n'a pas reproduit la rétention : tous les propriétaires étaient déjà libérés avant GC ou synchronisation, en readonly comme en writable. Preuve : `work/pr6-mlx-lifetime.txt`. Ce résultat écarte l'import MLX seul comme explication de cette reproduction ; il ne prouve pas tous les scénarios Metal du modèle réel.

Correction ciblée autorisée par le parent : parcours itératif dans `retained_size`, sans closure récursive. Même déduplication des propriétaires et mêmes charges pour containers imbriqués, `PackedMatrix` et objets backend. Trois tests avec GC cyclique désactivé prouvent une libération immédiate après inspection seule, rejet du cache et clear du cache. Un quatrième vérifie les charges imbriquées et le partage de backing buffers. La campagne réelle après correction relève du parent ; cette note ne prétend pas l'avoir validée.

## P2 — erreurs et validation du budget

- PR6 accepte des acquisitions négatives, booléennes et flottantes ; un montant négatif réduit `live_bytes` et permet de dépasser la capacité. `_release` peut également rendre le compteur négatif ou l'augmenter. Les méthodes privées sont maintenant validées (`type(size) is int`, montant >= 0, release <= live) avant mutation.
- PR6 libère la réservation après erreur de conversion alors que le traceback de la fonction appelée peut encore retenir le buffer fourni. La propriété et la charge sont désormais transférées au buffer avant la conversion NumPy fallible. La charge reste présente tant que le traceback conserve ce buffer, puis disparaît à sa libération.
- Sur short read, `_pread` de PR6 garde les bytes partiels dans son frame de traceback alors que l'opération relâche la réservation. Le résultat partiel est maintenant supprimé avant de lever l'erreur. Test par injection d'un short read de 5 octets.
- Les chemins d'erreur IO et l'admission concurrente ont été vérifiés sans double libération ; un second appel rejeté ne consomme ni read ni budget restant du premier.

## Pic de payload, metadata et limites

- `read_packed_rows(copy=False)` réserve un payload encodé E ; `copy=True` réserve 2E pour le read brut et la copie writable. La copie scratch brute est supprimée avant la sortie de réservation. Des tests inspectent la réservation déjà active au moment de `os.pread`.
- Un `owner_budget` externe compte les payloads retenus indépendamment de la réservation IO du lecteur. Ces pools ne constituent pas une mesure RSS et ne couvrent pas les allocations arbitraires faites par un appelant (`tobytes`, copie NumPy supplémentaire, etc.). Les métadonnées Python relèvent des estimations déjà documentées ; ces compteurs de payload ne sont pas une borne exacte de la mémoire processus ni du cache OS.
- Aucun bug supplémentaire reproduit dans `bulk_metadata`. Les 11 types scalaires donnent les mêmes valeurs et charges que le parseur scalaire. Un fixture avec seulement 16 octets de marge force des chunks de 16 octets et reste dans la réservation totale. Un fixture trop grand est rejeté avant la lecture des 8 000 octets de payload. Les tableaux de chaînes restent dans le chemin existant.
- Sélections vides vérifiées avec budget propriétaire nul, sans charge fantôme.
- Cet audit ne valide ni un modèle complet, ni le débit SSD physique, ni la performance Metal, ni les sorties de génération. Les essais intégrés restent nécessaires.

## Fichiers gelés après vérification

- `air_llm/airllm/nextgen/gguf.py` SHA-256 `300c7d03bdeea9a73ad9f1b8b965de39aaa1051f83500b52b6aa72af81ea8001`
- `air_llm/airllm/nextgen/adaptive_cache.py` SHA-256 `efdd1ab735df3ae33bc20df471a99e4d3e7302df4a53264cb50c78b19e74b98d`
- `tests_next/test_packed_io_safety.py` SHA-256 `5f2041343a9181389cc1bb0af1a0ea9acc93add1fba2ec26214d0c42ba828a1d`

`gguf_qwen.py`, les CLI et les nouveaux modules transport direct n'ont pas été modifiés par cet agent.
