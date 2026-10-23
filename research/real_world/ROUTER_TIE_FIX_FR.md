# Échec D2 : égalité au seuil du routeur et correction ciblée

La campagne réelle a été interrompue après **77 tokens de D2** par
`ValueError: Qwen router top-k boundary tie`. Le défaut est un arrêt conservateur
de l'implémentation AirLLM, pas une réponse incorrecte démontrée du modèle.
L'échec et le fragment restent des preuves distinctes des essais corrigés.

## Cause reproduite

Le replay de la version originale reproduit **77/77 IDs de sortie identiques**
avant l'erreur, couche 5, position de cache 561. Les entrées du routeur, ses
poids, logits et probabilités sont capturés et finis :
[`router-failure-replay.json`](../../evidence-real-world/router-failure-replay.json)
et la capture NPZ conservée localement sous
`evidence-real-world/router-failure-replay.npz` (non publiée : elle contient la
matrice de poids du routeur). Le JSON public décrit ses tableaux et son empreinte.
Le SHA-256 du NPZ est
`0b343cc940dfa5099fb87832acc1117998f3c95833658bf8b4a0b8095125cacd`.

Une comparaison supervisée portant exclusivement sur ce petit routeur donne :

| Calcul | Logit expert 49 | Logit expert 206 | Expert au huitième rang |
|---|---:|---:|---:|
| Gate MLX/Metal capturée | −4,415645599365234 | −4,415645599365234 | Égalité : arrêt original |
| Torch `topk` sur probabilités natives | Logits non recalculés | Logits non recalculés | 49 |
| Torch softmax/topk sur logits natifs | Identiques | Identiques | 49 |
| Gate Torch CPU complète à partir de x/W capturés | −4,415645599365234 | −4,415645122528076 | 206 |

Les probabilités natives des experts 49 et 206 valent toutes deux
`0.010838142596185207`. Après la gate Torch complète, elles deviennent
respectivement `0.010838143527507782` et `0.010838148184120655`.
L'erreur maximale entre logits du GPU natif et de Torch CPU vaut
`9.5367431640625e-7` sur cette capture. MLX CPU distingue également les deux
experts sur le même x/W.

L'hypothèse initiale d'une égalité créée uniquement par la softmax est donc
écartée pour cet incident : **les logits GPU sont déjà égaux**. Changer seulement
la précision de softmax ne peut pas départager des entrées identiques.
`precise=True` sur ces logits ne supprime effectivement pas l'égalité.
Départager simplement les probabilités natives avec Torch choisirait ici un
autre expert que le routeur CPU complet ; cette variante a été rejetée.

Preuve détaillée :
[`router-reference-probe.json`](../../evidence-real-world/router-reference-probe.json).
Versions mesurées : Torch **2.14.1**, MLX **0.32.4**. Cette probe CPU supervisée
s'est terminée avec `exit=0`, aucun dépassement, RSS échantillonné 237 Mo et
croissance de swap observée nulle. Aucun modèle complet n'a été chargé dans
Torch.

## Contrat officiel et périmètre exact

Le routeur officiel effectue `F.linear`, softmax FP32, `topk`, normalisation des
scores sélectionnés puis conversion vers le type des logits. Le code installé
de Transformers **5.20.2** est conservé avec son empreinte dans
[router_fix_contract.json](router_fix_contract.json). La même séquence figure
dans la [source officielle Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py).

La [documentation PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html)
ne garantit pas un ordre stable des indices ex æquo. La correction choisit donc
une référence locale explicitement versionnée : le routeur Torch CPU installé,
sur les x/W et le chunk courant. Elle ne prétend pas imposer un ordre universel
des égalités sur tous les backends ou toutes les versions de Torch.

## Correction livrée

- `qwen_route` garde sa politique générique `tie_policy='error'` par défaut.
- Le backend GGUF opte explicitement pour `torch_cpu_reference` en lui passant
  les budgets de workspace, de réserve système et ses compteurs.
- Sans égalité au seuil, calcul et sélection restent natifs. La softmax globale
  n'a pas été modifiée ; un test vérifie les résultats identiques bit à bit sur
  le chemin sans égalité.
- En cas d'égalité native, la gate complète est recalculée sur CPU pour **toutes
  les lignes du chunk aplati**, puis seuls les IDs et scores des lignes affectées
  sont réinjectés. Garder la forme complète évite de changer la forme de GEMM et
  le choix du kernel CPU en passant à un sous-batch.
- Aucun expert ni modèle complet n'est chargé dans Torch. La matrice de gate
  de cet incident mesure 256 × 2048 FP32, soit 2 Mio.

Les compteurs enregistrent nombre de vérifications/replis/lignes, version et
opération de référence, temps d'import, copie hôte, linear CPU, softmax CPU,
topk/normalisation, durée totale du repli et workspace estimé.

## Protections et vérifications

Le budget vérifie les tableaux du routeur déjà vivants et une marge conservatrice
pour les copies, conversions, scratch CPU et indices. Toutes les lignes du chunk
sont comptées. Avant la première importation de Torch, **384 Mio supplémentaires**
sont réservés à son runtime ; ce montant n'est pas présenté comme un plafond
d'allocation du runtime. La réserve système configurée est conservée. Un manque
de workspace, de réserve ou une référence indisponible produit une erreur
explicite, jamais une sélection arbitraire.

Les plafonds externes RSS, réserve, swap et pression mémoire ne sont pas abaissés.
Les coûts de l'import et de la lecture des bibliothèques doivent rester inclus
dans les temps/lectures process de la vraie requête corrigée.

**13 tests ont passé localement**, avec la capture présente, y compris :

- égalité vraie et égalité introduite par arrondi de softmax sur petites données ;
- politique générique conservant son refus explicite ;
- chemin sans égalité identique bit à bit et sans import Torch ;
- refus des budgets avant import et maintien de la réserve système ;
- batch mixte : gate CPU de forme complète, seules deux lignes sur trois réinjectées ;
- véritable capture GPU : même erreur sans opt-in, puis IDs **et scores** exactement
  égaux au routeur CPU complet de référence après correction, avec expert 206.

La dernière suite, incluant uniquement cette petite gate sur GPU, termine sans
dépassement : RSS échantillonné 245 Mo, croissance de swap nulle, `exit=0`.
Preuves publiques :
[`journal`](../../evidence-real-world/router-fix-tests-batch.log),
[`garde`](../../evidence-real-world/router-fix-tests-batch.guard.json).
Le XML détaillé reste local. Le NPZ n'est pas inclus dans le dépôt public ; la
régression réelle doit recevoir son chemin via `AIRLLM_TEST_ROUTER_CAPTURE` et
sa limite de taille est contrôlée avant lecture. Sans cette variable, ce seul
cas est explicitement ignoré : la validation ciblée publique donne **12 tests
réussis et 1 ignoré**, avec son motif visible dans
[`router-public-tests.log`](../../evidence-real-world/router-public-tests.log).
Ce réglage concerne uniquement la découverte de la fixture par le test ; le
moteur et ses empreintes ci-dessous restent inchangés. Les JSON et le programme
`benchmarks/replay_real_world_router.py` permettent de reproduire une capture
avec un checkpoint local conforme, sous supervision.

Ces tests valident le traitement de l'incident au niveau du routeur. Le replay
réel ci-dessous complète cette preuve. La correction n'établit pas une concordance
bit à bit de tout le modèle avec Torch.

## Replay D2 réel corrigé : arrêt résolu, interprétation de date à distinguer

Le tour **D2 est achevé avec 79 tokens et un EOS légitime** dans
[`campaign-resumed.json`](../../evidence-real-world/campaign-resumed.json).
Ses **77 premiers IDs sont exactement identiques** à ceux de l'essai interrompu
conservé dans
[`campaign-before-router-fix.json`](../../evidence-real-world/campaign-before-router-fix.json).
Les deux tokens supplémentaires sont le point final (`13`) puis `im_end`
(`248046`). Le prompt et le plafond de génération sont également identiques.

| Mesure du tour D2 achevé | Valeur observée |
|---|---:|
| Durée murale de la réponse | 281,5938 s |
| TTFT moteur | 189,8259 s |
| Replis du routeur / lignes affectées | 1 / 1 |
| Durée instrumentée du repli | 0,673150 s |
| Dont import de Torch | 0,658853 s |
| Dont gate CPU complète | 0,010025 s |
| Workspace estimé du repli | 8 433 228 octets |
| RSS maximal échantillonné du tour | 531 431 424 octets |
| Croissance maximale du swap système pendant ce tour | 0 octet |
| Mémoire disponible minimale échantillonnée | 3 928 178 688 octets |

L'import explique l'essentiel du coût **directement instrumenté** du repli.
La différence des durées globales entre l'essai interrompu, son replay de capture
et le tour achevé ne mesure pas causalement le coût du correctif : les états de
cache et les points d'arrêt diffèrent. Aucun gain ou surcoût global n'est déduit
de cette simple différence. Les maxima mémoire ci-dessus concernent le tour D2,
pas une certification de tous les tours ultérieurs de la campagne.

La réponse achevée indique toujours **« Date actuelle : 8 avril »**, alors que le
critère prévu visait la date d'événement, **22 mai**. Le 8 avril date bien la
version du document. La demande dit toutefois « date actuelle » sans préciser
explicitement « date actuelle de l'événement » : ce résultat est un écart au
critère prévu, mais **ne prouve pas à lui seul une erreur de lecture du modèle**.
L'égalité du préfixe confirme que cette interprétation figurait déjà dans le
texte avant le point de défaillance ; elle ne résulte pas de la continuation
corrigée. Un essai à formulation clarifiée doit être évalué séparément, sans
réécrire cette réponse originale ni prétendre que le correctif moteur change sa
qualité.

La réponse intégrale, les compteurs et les empreintes sont également figés par
tour dans [router_fix_contract.json](router_fix_contract.json). Le fichier
cumulatif de campagne peut encore recevoir des tours : son hash d'instantané
n'est pas présenté comme un hash final. Le hash canonique du tour D2 achevé est
`94ceba9a446dd0dfcea658f03c8820b04068008bfa5a69e4b8ebf20ac6910477` ; celui du
texte UTF-8 est `efe3afa03f4184ea738681fe3dd302627a9365708236e70e4f2ee928a7e2070b`.

Sources gelées de la correction :

- `qwen_moe.py` : `bc0ec17002407f1cbdfa523df38ae6a6358146d6eb11f8a0c573bd41ad61817e`
- `gguf_qwen.py` : `9e1b2e325ab296b7e27a9e88e7f5bc51236d7aebacd692ef7a9607120ca7ed99`
- `gguf_chat.py` : `6af7f468b1de6282eb366fd8918732bd5f95792e4a5bad1e5b2b8a11b5e15074`
- programme du replay avec sauvegarde atomique : `5b9d6f72a7ecb9c4e62917defcf99ae2b9aeade0ef197d831850cf4d4cba3e17`

Ces quatre empreintes déclarées dans l'identité du replay ont été comparées aux
sources relues lors de cette mise à jour ; elles concordent. Aucun fichier du
moteur ou du programme de campagne n'a été modifié pour cette mise à jour.

## Documentation primaire vérifiée en ligne

La documentation [PyTorch 2.14 — torch.topk](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html) ne garantit pas la stabilité des indices en cas de valeurs égales. Le contrat retenu reste donc celui du build CPU local mesuré, et non une identité universelle entre CPU, CUDA et Metal.

La documentation [MLX 0.32.4 — softmax](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.softmax.html) décrit `precise` comme un contrôle d’accumulation pour les entrées de moindre précision. Ici, les logits capturés sont déjà FP32 et leur égalité existe avant softmax : changer cette option ne peut pas reconstituer les différences perdues dans le produit matriciel. Cette conclusion vient de la capture et de la comparaison locale, pas de la seule documentation.
