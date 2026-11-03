# CPU16 : candidat rejeté, arrêt contrôlé

Le premier prompt gelé de 521 tokens se termine en 820,050836167 s. Les 154 880 logits finaux diffèrent du témoin CPU8 (écart maximal 1,61790836), ainsi que l’état final. Les quatre premiers tokens greedy sont identiques : cela ne suffit pas à qualifier la fidélité.

Les captures sont identiques jusqu’au token 511 et les checkpoints 128/208/256/384/512 concordent. Au token 512, la première divergence capturée est KDA couche 0 (6 839/8 192 valeurs, maximum 9,31e-10). Les routes changent d’ordre couche 4, puis d’ensemble au token 513 couche 3. Les indices de token, couche et rang sont ceux des traces natives, depuis zéro. La queue comporte 9 tokens en CPU16 contre 8+1 en CPU8. La bascule source nrc2/nrc1 est une hypothèse forte ; les projections précédant KDA ne sont pas capturées.

Le contrôleur a vérifié les fichiers bruts puis signalé uniquement son superviseur possédé. Le wrapper conserve `cancelled`, le natif `-15`, le superviseur `failed`. Les contrôles secteur/intégrité passent ; aucune violation mémoire ne déclenche cet arrêt. Le bras reste incomplet, le second prompt n’ayant pas fini, et aucun gain n’est retenu.

Les treize fichiers gzip sont identiques aux originaux après décompression ; `selection.json` donne tailles et SHA. Aucun poids ni payload numérique volumineux n’est publié.
