# Clôture indépendante du C16 original

Source préparée seulement : `audit_agents_cpu16.py` n'a été ni importé, ni exécuté,
ni testé, ni passé à AST pendant C16. Aucun fichier exécuté n'a été modifié.

Le lanceur figé accepte plusieurs séquences, efforts et tailles de batch. Son
indicateur `functional_two_agents_qualified` ne suffit donc pas à démontrer le
profil précis demandé. L'audit exige séparément mode agents, séquence 1→600→1,
16/16, CPU10/GPU0, slot unique, budget 512/2700/7200, admission effective de16 et
argv OS identique à la commande construite, avec tous les caches désactivés.

Le `reasoning_effort=null` observé correspond au défaut natif, pas à une preuve
automatique de Max. Le template GGUF exact choisit Max hors low/high ; le chemin
Engine→stream_chat n'injecte aucun override lorsque l'effort est null. L'audit
exige donc aussi chaque `template_request` sans override ou explicitement max,
et le marqueur natif `Reasoning Effort: Max` dans la section système rendue. Les
messages transmis à ce rendu doivent être ceux de la décision enregistrée.

Le contrat de complétion original autorise aussi une sortie au budget pour les
mesures de préfill, mais le mode agents impose EOS naturel. L'audit recompute
depuis les événements SSE bruts : terminal unique exact, stop eos, contexte non
tronqué, IDs entiers complets égaux au compteur final, sans tolérance EOS−1,
texte concaténé et résultat natif exacts. Les vérifications d'utilité originales
sont recopiées sans import runtime : final après `</think>`, bon repère privé,
absence du repère pair, calculs attendus (408/12 ; 131/142/11), justification et
conclusion affirmée correcte. Le lecteur doit aussi examiner les trois réponses
brutes : ces heuristiques ne constituent pas une preuve universelle de sens.

Le trou principal du `campaign_checks` original est le retour à l'agent1 : même
mission et marqueurs isolés ne prouvent pas la présence du premier échange. Le
runtime rejoue exactement les tours completed/partial du couple agent/mission,
mais le lanceur n'en teste pas l'égalité. L'audit ferme ce point : première
décision completed/EOS, mêmes tokens natifs, puis troisième `messages` composé
du système, du user initial exact, de l'assistant initial stocké exact, et du
nouveau user. La troisième requête `/apply-template` doit avoir reçu exactement
ces messages. Aucune égalité naïve du système entier ni du texte SSE brut avec
l'assistant n'est imposée : RONIN expurge cumulativement les secrets. L'égalité
correcte est `stored_output == redact(concat(SSE0))`, puis assistant rejoué égal
à ce stored_output. Les IDs ne sont jamais expurgés. Le rendu natif peut aussi
reformater les délimiteurs de raisonnement selon le template ; c'est l'entrée
structurée exacte au rendu qui établit le transport de la paire.

L'isolation exige l'absence du repère agent600 dans tous les messages, la mémoire
du retour1, le rendu natif et la sortie. La base SQLite close est vérifiée pour
600 IDs uniques exactement1..600 et pour les trois décisions 1→600→1. L'audit ne
connecte jamais SQLite au fichier original : il copie DB/WAL/SHM clos dans un
répertoire temporaire, puis interroge uniquement cette copie en mode lecture
seule. Aucun secret, environnement, clé API ou poids n'est lu. Les résultats ne
réexportent aucun prompt ni texte brut, seulement des verdicts et empreintes.

La preuve passive déjà conservée par root,
`agents-cpu16-live-observation.json`, lie PID88931, birth1793731641.671804,
exécutable et argv16/16 aux mêmes path/device/inode du verrou UID ouverts chez le
parent et le natif. Elle est référencée par SHA dans l'audit. C'est une observation
tardive : aucune attestation before-GO ou de possession flock depuis le démarrage
n'est déduite. La source `pass_fds` reste une preuve séparée d'intention.

Limite de ressource explicite : le runner original ne persiste pas directement
`NativeGlmModel._resource_baseline`, et son historique de samples est un deque
borné à3600. Une reconstruction exacte de toute la politique de swap à partir
d'une prétendue première baseline serait incorrecte. L'audit maintient les caps
8,5 GiB/32 MiB, réserve 768 MiB/headroom 512 MiB et grâces 3/20 s ; vérifie la
finitude, le cap RSS et les niveaux 1/2 des samples retenus ; conserve l'autorité de la
garde native originale, ses échecs persistants et la sortie normale. Le pic de
croissance de swap depuis la vraie baseline native reste null s'il n'est pas
observé. Le relevé passif de root fournit séparément une croissance depuis
`batch_admission`, qui n'est pas silencieusement assimilée à cette baseline.

Une réussite future du script est `closed_evidence_passed`, avec qualification
globale/numérique/performance toujours false. Elle concerne deux identités
observées et le retour1 parmi 600 identités enregistrées, pas600 générations ni
la stabilité universelle. Une clôture par timeout, limite, swap, annulation ou
historique absent reste rejetée ; aucune décision ni réponse n'est fabriquée.

Commande future, seulement après clôture et coordination root :

```text
/Users/macbook/Documents/Codex/2026-11-02/files-pasted-by-the-user-mission/work/venv/bin/python /Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/audit_agents_cpu16.py --output /Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/agents-cpu16-closure-audit.json
```

Correctifs de contre-revue source : la map de neuf artefacts de metadata est
exactement épinglée (chemins et SHA hex), vérifiée avant toute lecture déclarée.
Une map absente/incomplète/arbitraire est refusée. Les durées observées doivent
être finies : le temps natif last_request est comparé au budget 2700s ; les
temps runtime/capture, plus larges, restent nommés séparément. Le temps global
report.elapsed_seconds inclut le cleanup : son respect de7200 est rapporté sans
inventer un instant exact de fin de la fenêtre surveillée. Le moniteur original
reste autoritaire. Aucun test/AST/import n'a été exécuté pour ces corrections.

Dernier correctif : le chemin et le SHA attendus de l'exécutable doivent égaler
le pin exact AVANT toute lecture de runtime ; une config arbitraire ne peut donc
faire hacher un poids ou fichier personnel. Deux champs descriptifs sont séparés
du gate strict : first_two_useful_eos_observed conserve les deux EOS utiles1/600
prouvés même si la troisième requête échoue ; full_sequence_history_audited exige
les trois décisions complètes et la preuve exacte du retour1. Aucune réussite
globale n'est déduite du seul préfixe. Tant que le run est actif/nonclos, ces deux
champs sont inconnus (null), et aucun DB/SSE n'est inspecté par le script.

Le troisième échec peut légitimement ne laisser aucun snapshot `row3.decision`
ni fichier SSE si le flux n'a pas commencé. Ces absences produisent des checks
false, jamais un tour reconstruit depuis la DB. La paire du retour utilise des
getters prudents ; le descriptif des deux premiers EOS complets est aussi
préservé en cas d'autre preuve globale invalide, avec une portée raw explicite
et le gate C global toujours false. Aucun essai n'a été lancé.
