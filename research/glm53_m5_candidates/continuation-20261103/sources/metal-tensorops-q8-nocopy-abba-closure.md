# Clôture ABBA NoCopy — essai rejeté

Le run1 s’est arrêté au premier CPU A1 (M8192/K4096/N8), avant tout oracle
CPU validé ou comparaison. RSS enfant observé : 136 773 632 octets
(130,4375 Mio), au-dessus du plafond inchangé de 134 217 728 octets
(128 Mio). Durée contrôlée : 4,854266375 s. Aucune commande GPU, aucun
fichier F32 final et aucune comparaison ABBA achevée. Le pic RSS natif
de durée de vie n’a pas été produit ; il reste indisponible, pas nul.

Le seul enfant possédé a reçu SIGTERM et a quitté avec le code −15.
L’assertion de veille s’est terminée naturellement, le FD UID parent est
fermé, secteur/continuité et identité des sources sont positifs. Les
95 pins ont été revérifiés. Le binaire et son vrai reçu sont ceux de la
préparation ; aucun rebuild, changement de plafond ou nouvelle tentative.

Les logs, échantillons et reçu brut sont conservés sous abba-two-shapes/run1.
Le reçu de clôture ajoute leurs tailles/SHA et les identités exactes. Les
deux gains NoCopy apparents restent non qualifiés : ce refus n’établit ni
une accélération GPU, ni un débit GLM, ni un coût SSD.
