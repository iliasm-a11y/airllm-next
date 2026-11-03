# Validité temporelle et état d'alimentation

Un cycle E8 a traversé une période Sleep/DarkWake : son temps UTC écoulé
était d'environ 5 912 s pour environ 163 s au compteur de performance. Ses
durées de décodage ne qualifient donc pas une amélioration de débit. Une preuve
mathématique de fidélité reste distincte d'une mesure temporelle valide.

Le nouveau module `power_state.py` ne modifie aucun réglage permanent. Les
admissions du benchmark et du plan de campagne exigent un capot **ouvert** et
une source connue, secteur ou batterie. La source du plan est imposée à tous
les stages et revérifiée avant et après chacun ; un changement refuse la mesure.
La batterie n'est pas exclue par principe.

Chaque échantillon du moniteur compare `time.time()` à `time.perf_counter()`
depuis son point initial. Un retour en arrière de l'une des horloges ou un
désaccord cumulé supérieur à deux secondes invalide l'exécution. L'état du
capot et la source sont relus toutes les quatre itérations du moniteur, ainsi
qu'aux admissions et dans les chemins de finalisation. Les métadonnées conservent
UTC, les deux compteurs, leur désaccord et le motif d'échec. Ce contrôle détecte
un désaccord compatible avec la veille ou un changement d'horloge ; il ne
prétend pas identifier sa cause physique avec certitude.

Un processus possédé `/usr/bin/caffeinate -i -w <PID serveur>` demande seulement
la prévention de la veille d'inactivité pendant la vie du serveur possédé.
L'identité des deux PID est vérifiée avec leur date de création ; le helper est
surveillé, puis arrêté et attendu dans sa portée de durée de vie. Il n'utilise
ni `-s`, ni changement `pmset`, ni contournement de fermeture du capot. Son
identité, sa commande et sa libération doivent figurer dans la preuve de mesure.
Une sortie naturelle 0 n'est acceptée que lorsque le serveur surveillé est déjà
terminé ; une sortie -15 requiert un SIGTERM enregistré du propriétaire. Une
sortie précoce 0, une erreur, un signal externe ou un nettoyage forcé SIGKILL
invalide la qualification, même si le helper a bien été récupéré.
Le helper ne remplace pas les gardes mémoire et processus.

Les requêtes `ioreg` et `pmset` utilisent des arguments structurés sans shell,
une limite de sortie de 128 Kio et un délai de trois secondes. Un plist inconnu,
une valeur de capot non booléenne ou une source ambiguë refuse l'admission.
Les observations sont échantillonnées et non atomiques : elles ne démontrent
pas chaque événement matériel entre deux lectures.

Les [fixtures portables](test_power_state.py) couvrent capot fermé/ouvert/inconnu,
secteur/batterie, erreur de lecture, veille simulée, horloge inversée, dérive
cumulée, finalisation, changement de source, identité des PID et nettoyage sur
exception. Avec les tests de campagne et de complétion : **43 tests réussis**.
Aucune compilation native, aucun modèle, aucun GPU et aucune action de réveil
n'ont été lancés pour ces tests. Le [smoke réel](evidence/power-preflight-closed.json)
a observé le capot fermé et la batterie : le benchmark a été refusé avant toute
création de serveur. Les anciennes campagnes sans ces enregistrements restent
sans preuve rétroactive de continuité d'alimentation/horloge.
