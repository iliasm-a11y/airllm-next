# Q8 CPU : la géométrie des colonnes change les bits

Diagnostic clos à 2026-11-03T18:14:17.188552+00:00 : sur M8192/K4096 Q8_0, la partition
proposée v4 est bit à bit identique à8+reste dans42/42 cas et19,791,872 sorties
F32. Le backend direct CPU10 garde use_ref=false et le vrai trait vec_dot.nrows=2.
Les poids sont quantifiés ligne par ligne avec la routine CPU native. Aucun
scheduler, BLAS, modèle ou GPU n'a été initialisé par ce composant.

fullN diffère de8+reste dans20/42 cas : N=18,22,26,30,38,44,72,88,104,120, avec les deux
graines. Total7 464 822 F32 différents, maxabs3,0994415283203125e-6 ; toutes les
sorties sont finies. Ce sont des écarts du contrat bit à bit malgré leur faible
amplitude locale. Le source CPU utilise ceil(N/16) puis ceil(N/nchunks) : N30
donne deux plages15, qui imposent nrc1 ;16+14 conserve nrc2. Les champs nrc sont
explicitement dérivés du source et des traits, sans instrumentation des appels.
Le résultat est cohérent avec ce mécanisme ; aucun trait/kernel n'a été modifié.

Build0 en1.282035s ; diagnostic/contrôleur0 en3.769604s,
fenêtre superviseur4.392465s. getrusage natif : pic RSS à vie
117522432B, sous128MiB. Pic échantillonné du PID :
106201088B ; somme échantillonnée de l'arbre
externe :121683968B (mesure distincte,
avec possible double compte de pages). Croissance de swap0, aucune violation,
secteur admis ; les plafonds originaux8,5GiB/32MiB/headroom sont conservés.

FD UID observé dans le PID/birth/exe exact avant GO ; natif et caffeinate sortis
naturellement0, aucun signal envoyé, superviseur0 et UID libéré. Entrées suivies
inchangées. Cette synthèse lit uniquement les reçus clos, sans rerun.

Aucun F32 exporté : bitcounts/maxabs/SHA sont calculés dans le natif, sans
recomparaison indépendante depuis des raws. Le test synthétique résident ne
qualifie ni GLM complet, ni KDA/routes/logits/EOS, ni performance/cache/I/O SSD.
Il soutient la barrière de forme v4 avant le modèle strict, sans la remplacer.

SHA256 :

- CPP `66ae95a60f9d1f061a8d3700ce7cbe7f676a5220fbabf133b61bfda650ee9527`
- Binaire `d7274e60015b35a8696f4e21acc79c4172eb693bc10a3715651aaa877a9d6b0a`
- Reçu build `57d3f20a11958daedce099ab148029e26ec979263728e2ed3c06d0fe0725ff50`
- Reçu diagnostic `d559956bd5b79d679e514604c5db58a90a117054dd332364038d40886f1992e4`
- Reçu garde `2c2660ebf737de0144dcc7c7da220d1147c95b4d848e9601e50168744c66b324`

Inventaire et valeurs exactes : `mechanism-summary.json`.
