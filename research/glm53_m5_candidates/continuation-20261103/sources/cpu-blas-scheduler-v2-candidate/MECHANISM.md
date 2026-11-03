# Mécanisme CPU8 → BLAS32 confirmé sur composant réduit

Le scheduler natif [BLAS,CPU], avec `op_offload=false` et de vrais poids
CPU_Mapped marqués WEIGHTS, choisit CPU àN8/N16 et BLAS àN32. Sur deux seeds
non triviales, lesN8/N16 sont bit à bit identiques au témoin CPU8. ÀN32, les
4096 sorties sur4096 diffèrent pour chaque seed, dès l'indice0 ; erreurs absolues
maximales0,015421104617416859 et0,016957998275756836. Le CPU directfull32 reste
bit à bit identique au CPU8 dans les mêmes cas. Toutes les valeurs sont finies.

La différence source est substantielle : BLAS déquantifie les poids versF32 puis
emploie les activations F32 originales, tandis que le chemin CPU Q8 quantifie
les activations enQ8_0. Le seuil min_batch32 est dans ggml-blas.cpp396–432.
Le modèle ajoute les backends ACCEL avantCPU dans llama-context.cpp505 ; le
scheduler peut choisir BLAS depuis des poids host malgré op_offload=false.
Ce mécanisme explique une possibilité réelle de divergence préKDA ; il ne
prouve pas encore la causalité sur les captures GLM complètes.

Le test réduit est M128/K4096, N8/16/32, deux seeds. La matrice BLAS déquantifiée
occupe2Mio ; M8192 demanderait128Mio avant les autres allocations et dépasserait
le cap128Mio du composant. Aucun test de cette grande déquantification n'a été
fait. Aucun modèle, GPU, routage, KDA, EOS ou gain global n'est qualifié.

Le natif a terminé0, sans signal ; son getrusage lifetimeRSS vaut
12828672 octets. Le contrôleur a duré
0.498088625s, le guard externe
1.137941708s, avec swap échantillonné supplémentaire
0 octet. Le FD UID exact a été
observé dans le natif avant GO, après vérification PID/birth/exécutable et
assertion anti-veille. Le caffeinate natif s'est fermé naturellement0 ; UID
externe libéré, secteur/couvercle/continuité et empreintes inchangées.
Ces temps décrivent la clôture du diagnostic ; ils ne mesurent pas un gain.

Binaire : `59a846c8d42a06a65e28e201a89af0067e56e9eb0a071b9519801a35d7f3f32d`. Reçu natif/contrôleur :
`53ff5fbecaffe5de6d03a08df75c07ff2c62e03b04f8020212111e738fa22c7b`. Guard externe :
`56acc5b3c0d083334d2e4f3ad407cea175b2e4cf5ed0edd8aa34e2689ead423e`. Les six cas et toutes les empreintes
sources/builds figurent dans `mechanism-summary.json`.

L'API publique `ggml_backend_unload(regBLAS)` est disponible à backend.h262.
Son implémentation reg.cpp266 retire les devices puis le registre. Un helper
expérimental distinct peut l'appeler avant toute création de modèle/contexte,
puis vérifier l'absence de BLAS. Aucun changement de bibliothèque n'est requis ;
sa fidélité GLM devra être mesurée séparément sous une nouvelle identité.
