# Réponses réelles, identiques entre variantes

## short

Prompt : Cite les trois couleurs du drapeau français, sans autre explication.

Réponse intégrale réellement générée :

Bleu, blanc, rouge

## explanation

Prompt : Explique en français, en 100 mots maximum, comment un modèle d'intelligence artificielle de 24 Go peut fonctionner sur un ordinateur possédant seulement 16 Go de mémoire vive.

Réponse intégrale réellement générée :

Un modèle de 24 Go peut fonctionner sur 16 Go de RAM grâce au **swap** (mémoire virtuelle). Le système d’exploitation déplace les données inactives du modèle depuis la RAM vers le disque dur (SSD/HDD), libérant ainsi de l’espace mémoire. Seules les parties du modèle activement utilisées restent en RAM. Bien que cette méthode permette l’exécution, elle est **beaucoup plus lente** car l’accès au disque est bien plus lent que celui à la RAM. De plus, une partie de la mémoire est réservée au système et aux autres applications, ce qui peut causer des ralentissements ou des erreurs si la mémoire totale (RAM + swap) est insuffisante.

## long_context

Prompt : Dossier technique fictif pour une expérience locale. L'ordinateur possède 16 Gio de mémoire unifiée et utilise son SSD interne. Le modèle est stocké dans un fichier GGUF de 23,94 Go. Le système peut lire sélectivement les poids nécessaires à chaque couche. Le cache de poids est limité à 64 Mio, le budget des entrées et sorties à 64 Mio, les états de génération à 256 Mio et l'espace de travail à 128 Mio. La réserve système de 3 Gio est indépendante de ces budgets. Le superviseur vérifie la pression mémoire et l'augmentation du swap toutes les 50 millisecondes. Aucune application personnelle n'est fermée. Le modèle conserve ses poids d'origine. Question : quelle est la somme des quatre budgets de cache, entrées et sorties, états et travail ? Réponds par une seule phrase, en Mio, sans inclure la réserve système.

Réponse intégrale réellement générée :

La somme des quatre budgets de cache, entrées et sorties, états et travail est de 512 Mio.
