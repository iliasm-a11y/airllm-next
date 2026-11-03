# Comparateur NoCopy 128 — lecture historique bornée v4

Cette copie de v3 corrige uniquement la lecture du garde CPU8 historique, qui
mesure 26 277 130 octets et dépassait la borne JSON générale de 16 Mio. Les
sources v3, son reçu INVALID avant numérique et les captures restent intacts.

L'exception de 32 Mio exige d'abord le chemin exact
`work/native-full8/guard.json`, la taille connue et le descripteur explicitement
épinglé dans le nouveau prepare manifest. Elle lit au maximum 32 Mio + 1 octet,
vérifie le SHA des mêmes octets ensuite parsés et leur identité de fichier :
`b991594498b14afc2a8b273da87843fb78e5534a023a053df393e2b6be0ecd08`.
Un autre chemin, une autre taille ou un autre contenu est refusé. Le candidat
et tous les autres JSON conservent leur borne de 16 Mio.

La nouvelle liaison de builds inclut le descripteur de ce garde historique et
le SHA du prepare v4. Elle doit être recréée ; la liaison v3 reste historique.
Le validateur du garde de build est copié sans modification. Ni les gardes
RSS/swap/alimentation, ni les clés mathématiques, ni les critères des 118
séries, des checkpoints ou les tolérances numériques ne changent.

Cette correction d'accès aux preuves ne produit aucun verdict numérique. Une
fenêtre128 reste un contrôle de rejet incomplet, sans qualification de prompt
complet, EOS, préfill complet, hors mémoire ou performance globale. Aucun modèle
ne doit être relancé pour appliquer cette correction hors ligne.

La commande exacte de nouvelle liaison et celle de comparaison des captures
closes sont dans `selection.json`. Root conserve la coordination de leur
exécution. La revue indépendante et la vérification AST/lecture seule sont
consignées séparément, sans remplacer une comparaison stricte réussie.
