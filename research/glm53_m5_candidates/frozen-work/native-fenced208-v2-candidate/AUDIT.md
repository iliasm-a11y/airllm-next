# CPU v2 : frontières globales explicites à 208

Candidat source isolé, non compilé et non exécuté. Le helper v1 actif, son
wrapper, ses bibliothèques, ses snapshots et les sources GPU figées ne sont
pas modifiés. Aucun résultat numérique v1 n'est attribué à ce nouveau binaire.

La politique `cpu8-tail-global-fences208-v2` borne chaque appel par le prochain
checkpoint absolu dans `[128,208,256,384,512]`, la fin de la phase et UB. Une
queue impaire supérieure à8 conserve son préfixe multiple de8 puis le reste
impair1/3/5/7 ; une queue impaire de taille≤8 reste entière. Toutes les fences
sont multiples de8. Le profil inscrit la liste exacte et la version2 dans
`state_parameters` : ils participent donc à une nouvelle clé complète. Les
descripteurs de chaque appel enregistrent la fin de phase, la fence retenue,
le lot avant fence et le lot final. Aucun entier supplémentaire n'est ajouté
aux prompts ou aux positions natives.

La frontière208 devient une vraie frontière physique dans le préfill full
pour UB8/16/32/64/128. Après cette frontière, full, split et restore utilisent
exactement le même découpage du suffixe privé. Le snapshot reste unique et
immuable, sous256Mio ; le budget de scratch16Mio demeure une politique
conservatrice, non une mesure. Les gardes, tailles d'état, sampler diagnostic,
captures et flags de sérialisation complets sont conservés.

| Prompt | UB | Appels v1 | Appels v2 |
|---:|---:|---:|---:|
| 521 | 8 /16 | 66 /34 | 66 /34 |
| 521 | 32 /64 /128 | 18 /10 /6 | 19 /11 /7 |
| 542 | 8 /16 | 68 /34 | 68 /34 |
| 542 | 32 /64 /128 | 17 /9 /5 | 18 /10 /6 |

Ces comptes proviennent du plan de découpage, pas d'une mesure de latence.
Une requête32+ gagne une frontière physique supplémentaire autour de208.
Elle peut relire des poids ou ajouter une synchronisation ; aucun coût en
secondes ni gain net de cache n'est déduit du seul nombre de tokens.
Le checkpoint208 supplémentaire32+ ajoute aussi une sérialisation/hachage
diagnostique d'état sous les admissions existantes. Il n'ajoute pas de copie
persistante par agent. Le pic réel et l'espace de captures restent à mesurer.

Le domaine GPU v1 reste inchangé. UB64 introduit un appel N48 et UB128 des
appels N80 et N48 ; le provider GPU actuel les refuse et ces calculs restent
CPU. Aucun nouveau noyau, domaine N ni rebuild du provider n'est demandé ici.
Un futur helper GPU v2 devra avoir sa propre source, son propre build et son
propre profil, même s'il réutilise les bibliothèques GPU déjà construites.

La propriété de parité teste seulement l'éligibilité fondée sur le nombre de
colonnes. Les heuristiques de chunks CPU, LLAMAFILE, dimensions des matrices,
routage MoE et états récurrents peuvent encore provoquer des divergences.
Le premier prompt exact du v1 UB16 ne prouve pas la fidélité du v2 UB32+.

`variant_compare.py` lie les deux builds CPU8 original et v2 réel, vérifie
sources/en-têtes/modules délégués et recalcule chaque clé propre. Son lecteur
de checkpoints exige208 pour chaque UB parce que les appels réels le
prouvent, et tous les multiples de128 prescrits par le helper jusqu'à la fin
du prompt borné. Il conserve les logits complets,118 noms, vues stridées,
routes ordonnées, écritures, états, continuations et replays bit à bit CPU.

`restore_compare.py` exige d'abord cette comparaison brute CPU8→fullV2.
Ensuite il compare fullV2→restoreV2 avec le même binaire, les mêmes paramètres
et la même clé v2. La condition statique `208 % UB == 0` est remplacée par
la preuve des fences des trois préfixes et des frontières physiques identiques
des suffixes, ainsi que le SHA exact de l'état208 et du snapshot. Tous les
autres critères du comparateur natif restent exigés. Aucun reçu mesuré n'est
retouché et aucune fonction existante n'est monkeypatchée.

La preuve reste un diagnostic borné. Annulation, reprise d'historique privé,
EOS utiles et performance de deux agents demandent leurs essais propres.
Un résultat partiel ou une simple géométrie correcte ne peut pas qualifier
le cache. Les empreintes du binaire et du reçu v2 restent nulles tant que le
coordinateur n'a pas réalisé un build distinct après la clôture du modèle actif.

Depuis `work/`, plan sans compiler :

```sh
PY=/Users/macbook/Documents/Codex/2026-10-27/mission-codex-installation-qwen-3-8/.venv/bin/python
"$PY" native-fenced208-v2-candidate/build_helper.py \
  --native-source metal-integration-v4-source/source \
  --libraries /Users/macbook/Documents/Codex/2026-11-02/referenced-chatgpt-conversation-this-is-an-2/work/runtime-v4/bin \
  --receipt native-fenced208-v2-candidate/build-plan.json
```

Le coordinateur pourra ensuite fournir un nouveau chemin de reçu et
`--execute`. Le binaire et la liaison seront alors calculés depuis le build
réel ; il ne faut pas réutiliser la liaison du helper v1.

Comparaison numérique d'un full v2 complet après build/exécution coordonnés :

```sh
"$PY" native-fenced208-v2-candidate/variant_compare.py \
  native-full8/capture "$FULL_V2/capture" \
  --reference-guard native-full8/guard.json --candidate-guard "$FULL_V2/guard.json" \
  --reference-build native-prefill-final-controls.build.json \
  --candidate-build native-fenced208-v2-candidate/build-receipt.json \
  --binding native-fenced208-v2-candidate/build-binding.json --output "$FIDELITY_V2"
```

Puis `restore_compare.py` accepte les mêmes chemins de build/liaison et
`--cpu-reference`, `--cpu-guard`, `--reference`/`--reference-guard` pour fullV2,
`--candidate`/`--candidate-guard` pour restoreV2 et `--output`. Il relit les
preuves brutes CPU8→fullV2 ; un JSON `passed:true` isolé ne remplace pas ces
captures.
