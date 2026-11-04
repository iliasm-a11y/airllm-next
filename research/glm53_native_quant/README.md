# GLM-5.3 Flash : dispatch quantifié natif, M5 / 16 Gio

**CPU32 contemporain terminé après nettoyage des outils inactifs ; kernels
Metal natifs réellement exécutés sur quatre composants originaux. Aucun
nouveau gain GPU de décodage complet n'est qualifié.** Base : PR #24, commit
`6cfa76930626520c15246286b3baa2a58c0cf952`.

Le [rapport français](RAPPORT-FR.md) distingue le code compilé, les tests
portables, l'audit des sources, les refus réels et les mesures historiques.
Les [preuves sélectionnées](evidence/selection.json) ne contiennent aucun
poids, vecteur d'activation ou état privé.

La reprise mesure 31 appels de décodage CPU en 191,954455206 s, soit
0,161496642 intervalle/s, préfill exclu, sans croissance du swap. Les kernels
Metal standards Q2_K/Q8_0 échouent aux tolérances de composant. L'adaptateur
d'activation les respecte sans égalité bit à bit, mais coûte 6,20 à 9,32 fois
le temps CPU10 synchronisé sur les deux formes et deux graines testées.
Les [preuves de reprise](evidence/recovery-20261104/selection.json) conservent
le nettoyage, CPU32, les quatre essais finaux et la série initiale dont le
temps CPU était biaisé par la recréation de threads.

## Ce qui est ajouté

- `audit.py` lit la révision llama.cpp épinglée avec `git show`, puis vérifie
  séparément les 13 sources V4 restaurées. Les extraits gardent leurs lignes
  et leur provenance ; un checkout modifié n'est pas assimilé à l'amont.
- `native.cpp` construit le même graphe `MUL_MAT` mono-colonne sur le backend
  CPU ou Metal de GGML. Il conserve les octets Q2_K/Q8_0 ; aucun nouveau
  kernel ni matrice de poids FP32 n'est créé.
- L'adaptateur expérimental quantifie **l'activation seulement** avec le
  codec temporaire CPU Q8_K ou Q8_0, puis la déquantifie en un vecteur FP32
  pour le kernel Metal existant. Il ne promet pas de reproduire l'ordre
  d'accumulation entier ARM : la comparaison doit décider.
- `probe.py` confronte CPU1, CPU10, Metal natif, cet adaptateur et le kernel
  historique `tile1` de PR #24 aux mêmes octets originaux. CPU1/CPU10 utilisent
  leurs pools successivement. Les modes Metal sont alternés aléatoirement.
  Le CPU, le Metal natif et le timer historique constituent des phases
  successives ; ce protocole n'est pas un A/B/B/A de modèle complet.
- Le banc rapporte les comparaisons bit à bit et les tolérances préexistantes
  `atol=1e-3, rtol=1e-4`, sans les élargir. Les activations par défaut sont
  synthétiques et explicitement identifiées. `--activation-f32` accepte un
  vecteur fourni, dont la provenance doit être décrite séparément.

Le temps total natif inclut préparation/import de l'activation, lancement,
attente de terminaison réelle et lecture de la sortie. Les sous-temps
`prepare_import`, `submit`, `terminal_wait` et `output_read` sont des durées
murales hôte ; `submit` peut déjà bloquer sur le backend CPU. Ils ne sont
pas des mesures isolées de temps GPU. Les journaux V4
`AIRLLM_GPU_TIME` sont conservés lorsqu'ils existent ; ils incluent chauffes
et deux modes, et ne doivent pas être arbitrairement attribués à un mode.
L'upload initial des poids et l'initialisation Metal sont rapportés à part.
Les poids des microtests restent chauds et résidents : aucun résultat de
composant ne vaut un gain sur le checkpoint complet de 110 Go.

## Reproduction

Depuis la racine du dépôt, avec Python, NumPy, psutil et pytest disponibles :

```sh
export PYTHONPATH=.:air_llm
python -m research.glm53_native_quant.audit \
  --source "$RONIN_LLAMA_SOURCE" --restored-v4 "$RONIN_V4_RESTORED" \
  --output /chemin/nouvel-audit

python -m research.glm53_native_quant.build \
  --source "$RONIN_LLAMA_SOURCE" --runtime "$RONIN_V4_RUNTIME" \
  --output /chemin/nouveau-build-natif

RONIN_CAMPAIGN_SWAP=$(python -c 'import psutil; print(psutil.swap_memory().used)')
python -m research.glm53_ultra_decode.model_profile \
  --model "$RONIN_MODEL" --receipt "$RONIN_CURRENT_RECEIPT" \
  --runtime "$RONIN_V4_RUNTIME" --probe "$RONIN_PR24_BUILD/native-decode" \
  --output /chemin/nouveau-temoin32 --tokens 32 \
  --campaign-swap-used-bytes "$RONIN_CAMPAIGN_SWAP"

python -m research.glm53_native_quant.probe \
  --model "$RONIN_MODEL" --receipt "$RONIN_CURRENT_RECEIPT" \
  --runtime "$RONIN_V4_RUNTIME" --build /chemin/nouveau-build-natif \
  --legacy-build "$RONIN_PR24_BUILD" --output /chemin/nouveau-composant-Q2 \
  --format 2 --tensor blk.40.ffn_gate_exps.weight --rows 2048 \
  --campaign-swap-used-bytes "$RONIN_CAMPAIGN_SWAP"

python -m pytest --noconftest --capture=sys -q \
  research/glm53_native_quant research/glm53_ultra_decode
```

Ne pas poursuivre une campagne de modèle après un refus ou un arrêt. Le
composant isolé a une admission différente, justifiée par son petit payload ;
son éventuel succès ne remplace jamais l'admission du modèle complet.
Pour Q8 : `--format 8 --tensor blk.3.attn_q_a.weight --rows 1536`.
Répéter avec une seconde graine et de nouveaux dossiers de sortie. Les deux
graines restent deux activations de composant, pas deux trajectoires GLM.

La compilation exige les bibliothèques V4 exactement identifiées dans le
reçu PR #24. Les binaires reconstruits ne récupèrent aucune qualification
historique du modèle. Le timer historique et ses entrées sont eux aussi
vérifiés avant/après ; un autre shader est refusé.

## Protections et suite conditionnelle

Le modèle conserve admission 5,5 Gio, réserve disque 32 Gio, RSS 8,5 Gio,
croissance du swap global 32 Mio, pression normale, checkpoint vérifié,
secteur/capot/horloges et verrou UID. Les composants réservent 3 Gio au
système et admettent 256 Mio de payload, conformément au profil précédent.
Une sélection ne lit pas plus de 16 Mio de poids ; les buffers GGML sont
comptés avant allocation. La surveillance externe englobe le worker et ses
descendants, avec budget cumulatif de swap et nettoyage des seuls processus
détenus. Les crêtes entre échantillons peuvent rester invisibles.

Le verrou du worker précède tout import de tableaux ou appel natif. La
préobservation d'un verrou libre n'est pas une réservation : une prise du
verrou par un autre moteur entre ces deux étapes est également refusée.
Les temporaires de poids du timer historique restent dans un dossier privé
de l'essai et sont supprimés à sa fermeture normale. Après arrêt forcé, ils
peuvent subsister localement ; la sélection publique ne les inclut jamais.

Un composant fidèle et avantageux justifiera seulement un essai d'intégration
GLM. Il faudra ensuite comparer logits complets, routes ordonnées, routeurs,
KDA et écritures d'état sur 16/32 étapes, puis 64/128 et EOS si admissibles,
avec une référence contemporaine et des essais alternés. Le cache MoE
inter-agents reste différé ; le pipeline attend également une mémoire stable.
Le moteur partagé RONIN conserve 600 identités, un modèle et un slot physique.
