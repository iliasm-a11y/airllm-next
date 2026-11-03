# C16 original : essai clos, refus fonctionnel

Le premier agent a reçu 523 tokens et produit 245 IDs partiels avant
`REQUEST_TIMEOUT`. Le préfill observé par root dans le log natif dure
818,994569 s. Aucun `</think>`, réponse finale utile ou EOS naturel n'est
observé. L'agent 600 et le retour à l'agent 1 n'ont pas été lancés. La campagne
dure 2719,097061 s, nettoyage compris ; aucune qualification C n'est acquise.

Le natif termine avec le code 0 lors de l'arrêt propre. Ce code ne prouve pas
un EOS. Son pic RSS échantillonné maintenu est de 7 155 351 552 octets, sous
le plafond de 9 126 805 504 octets. Les 2685 relevés conservés restent des
échantillons ; le watchdog natif rapporte `failure=null`, pression finale 1
et croissance swap finale 0. Le pic de croissance depuis sa véritable baseline
n'est pas reconstruit par l'audit. La continuité d'alimentation est validée.

Le reçu de nettoyage confirme l'absence des trois PID 88931/88910/88921.
L'assertion idle a été arrêtée par son propriétaire, code −15, avec le verdict
`owner_term` admis par sa politique. Aucun nouveau contrôle OS n'a été lancé
pour ce résumé.

L'audit v1 avait mal lu les ressources : `engine_after_stream.native` n'est
écrit qu'après les trois requêtes réussies. La révision v2 utilise les vrais
snapshots `engine_after_load`, `engine_before_cleanup`, `engine_after_cleanup`
pour les ressources et conserve le snapshot de stream obligatoire pour le
critère fonctionnel complet. L'ancien audit rejeté et les preuves restent
intacts. La révision v2 est préparée, non exécutée à ce stade.
