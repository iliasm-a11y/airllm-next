# Réponses intégrales — campagne 5.1

Campagne complète et appariée. Tokens, logits, routes et positions concordent par paire. Les textes ci-dessous sont les sorties enregistrées, sans réécriture. Le token EOS figure dans les IDs et dans le texte brut.

## explanation.r0.candidate

Commit moteur : `cdd426e1670a95c1798e45962c48e8d7ae87daaf`. Arrêt : EOS.

### Prompt intégral

```text
Explique en français, en 100 mots maximum, comment un modèle d'intelligence artificielle de 24 Go peut fonctionner sur un ordinateur possédant seulement 16 Go de mémoire vive.
```

### Réponse affichée intégrale

```text
Un modèle de 24 Go peut fonctionner sur 16 Go de RAM grâce au **swap** (mémoire virtuelle). Le système d’exploitation déplace les données inactives du modèle depuis la RAM vers le disque dur (SSD/HDD), libérant ainsi de l’espace mémoire. Seules les parties du modèle activement utilisées restent en RAM. Bien que cette méthode permette l’exécution, elle est **beaucoup plus lente** car l’accès au disque est bien plus lent que celui à la RAM. De plus, une partie de la mémoire est réservée au système et aux autres applications, ce qui peut causer des ralentissements ou des erreurs si la mémoire totale (RAM + swap) est insuffisante.
```

### Sortie brute intégrale

```text
Un modèle de 24 Go peut fonctionner sur 16 Go de RAM grâce au **swap** (mémoire virtuelle). Le système d’exploitation déplace les données inactives du modèle depuis la RAM vers le disque dur (SSD/HDD), libérant ainsi de l’espace mémoire. Seules les parties du modèle activement utilisées restent en RAM. Bien que cette méthode permette l’exécution, elle est **beaucoup plus lente** car l’accès au disque est bien plus lent que celui à la RAM. De plus, une partie de la mémoire est réservée au système et aux autres applications, ce qui peut causer des ralentissements ou des erreurs si la mémoire totale (RAM + swap) est insuffisante.<|im_end|>
```

### IDs générés, EOS inclus

```json
[1743, 79653, 401, 220, 17, 19, 5805, 23889, 237805, 1668, 220, 16, 21, 5805, 401, 21360, 73721, 7657, 2972, 24921, 332, 318, 155769, 93064, 224456, 553, 1900, 69073, 293, 515, 193400, 7204, 1940, 3424, 44064, 303, 519, 1821, 3717, 79653, 38767, 1147, 21360, 5266, 501, 204928, 10336, 318, 1178, 35, 22629, 3967, 681, 2953, 185536, 41825, 401, 324, 86056, 1260, 184743, 13, 1297, 2346, 3424, 9389, 3717, 79653, 3969, 1078, 211400, 205408, 644, 21360, 13, 71056, 1649, 19450, 94187, 176761, 324, 90032, 166568, 11, 25606, 1725, 2972, 1323, 43391, 5346, 194456, 332, 1740, 324, 84725, 11797, 7657, 204928, 1725, 13958, 5346, 46213, 1649, 62793, 3656, 1147, 21360, 13, 1526, 5346, 11, 5992, 46207, 401, 1147, 184743, 1725, 156171, 7639, 7657, 69073, 1778, 9750, 43503, 8098, 11, 3633, 7528, 23889, 2093, 846, 909, 240219, 160867, 5724, 909, 227128, 4259, 1147, 184743, 173275, 318, 32784, 478, 13881, 8, 1725, 220168, 284, 4783, 13, 248046]
```

## explanation.r0.baseline

Commit moteur : `ab59677fed6181d909866364a50dd07244af6b1b`. Arrêt : EOS.

### Prompt intégral

```text
Explique en français, en 100 mots maximum, comment un modèle d'intelligence artificielle de 24 Go peut fonctionner sur un ordinateur possédant seulement 16 Go de mémoire vive.
```

### Réponse affichée intégrale

```text
Un modèle de 24 Go peut fonctionner sur 16 Go de RAM grâce au **swap** (mémoire virtuelle). Le système d’exploitation déplace les données inactives du modèle depuis la RAM vers le disque dur (SSD/HDD), libérant ainsi de l’espace mémoire. Seules les parties du modèle activement utilisées restent en RAM. Bien que cette méthode permette l’exécution, elle est **beaucoup plus lente** car l’accès au disque est bien plus lent que celui à la RAM. De plus, une partie de la mémoire est réservée au système et aux autres applications, ce qui peut causer des ralentissements ou des erreurs si la mémoire totale (RAM + swap) est insuffisante.
```

### Sortie brute intégrale

```text
Un modèle de 24 Go peut fonctionner sur 16 Go de RAM grâce au **swap** (mémoire virtuelle). Le système d’exploitation déplace les données inactives du modèle depuis la RAM vers le disque dur (SSD/HDD), libérant ainsi de l’espace mémoire. Seules les parties du modèle activement utilisées restent en RAM. Bien que cette méthode permette l’exécution, elle est **beaucoup plus lente** car l’accès au disque est bien plus lent que celui à la RAM. De plus, une partie de la mémoire est réservée au système et aux autres applications, ce qui peut causer des ralentissements ou des erreurs si la mémoire totale (RAM + swap) est insuffisante.<|im_end|>
```

### IDs générés, EOS inclus

```json
[1743, 79653, 401, 220, 17, 19, 5805, 23889, 237805, 1668, 220, 16, 21, 5805, 401, 21360, 73721, 7657, 2972, 24921, 332, 318, 155769, 93064, 224456, 553, 1900, 69073, 293, 515, 193400, 7204, 1940, 3424, 44064, 303, 519, 1821, 3717, 79653, 38767, 1147, 21360, 5266, 501, 204928, 10336, 318, 1178, 35, 22629, 3967, 681, 2953, 185536, 41825, 401, 324, 86056, 1260, 184743, 13, 1297, 2346, 3424, 9389, 3717, 79653, 3969, 1078, 211400, 205408, 644, 21360, 13, 71056, 1649, 19450, 94187, 176761, 324, 90032, 166568, 11, 25606, 1725, 2972, 1323, 43391, 5346, 194456, 332, 1740, 324, 84725, 11797, 7657, 204928, 1725, 13958, 5346, 46213, 1649, 62793, 3656, 1147, 21360, 13, 1526, 5346, 11, 5992, 46207, 401, 1147, 184743, 1725, 156171, 7639, 7657, 69073, 1778, 9750, 43503, 8098, 11, 3633, 7528, 23889, 2093, 846, 909, 240219, 160867, 5724, 909, 227128, 4259, 1147, 184743, 173275, 318, 32784, 478, 13881, 8, 1725, 220168, 284, 4783, 13, 248046]
```

## explanation.r1.baseline

Commit moteur : `ab59677fed6181d909866364a50dd07244af6b1b`. Arrêt : EOS.

### Prompt intégral

```text
Explique en français, en 100 mots maximum, comment un modèle d'intelligence artificielle de 24 Go peut fonctionner sur un ordinateur possédant seulement 16 Go de mémoire vive.
```

### Réponse affichée intégrale

```text
Un modèle de 24 Go peut fonctionner sur 16 Go de RAM grâce au **swap** (mémoire virtuelle). Le système d’exploitation déplace les données inactives du modèle depuis la RAM vers le disque dur (SSD/HDD), libérant ainsi de l’espace mémoire. Seules les parties du modèle activement utilisées restent en RAM. Bien que cette méthode permette l’exécution, elle est **beaucoup plus lente** car l’accès au disque est bien plus lent que celui à la RAM. De plus, une partie de la mémoire est réservée au système et aux autres applications, ce qui peut causer des ralentissements ou des erreurs si la mémoire totale (RAM + swap) est insuffisante.
```

### Sortie brute intégrale

```text
Un modèle de 24 Go peut fonctionner sur 16 Go de RAM grâce au **swap** (mémoire virtuelle). Le système d’exploitation déplace les données inactives du modèle depuis la RAM vers le disque dur (SSD/HDD), libérant ainsi de l’espace mémoire. Seules les parties du modèle activement utilisées restent en RAM. Bien que cette méthode permette l’exécution, elle est **beaucoup plus lente** car l’accès au disque est bien plus lent que celui à la RAM. De plus, une partie de la mémoire est réservée au système et aux autres applications, ce qui peut causer des ralentissements ou des erreurs si la mémoire totale (RAM + swap) est insuffisante.<|im_end|>
```

### IDs générés, EOS inclus

```json
[1743, 79653, 401, 220, 17, 19, 5805, 23889, 237805, 1668, 220, 16, 21, 5805, 401, 21360, 73721, 7657, 2972, 24921, 332, 318, 155769, 93064, 224456, 553, 1900, 69073, 293, 515, 193400, 7204, 1940, 3424, 44064, 303, 519, 1821, 3717, 79653, 38767, 1147, 21360, 5266, 501, 204928, 10336, 318, 1178, 35, 22629, 3967, 681, 2953, 185536, 41825, 401, 324, 86056, 1260, 184743, 13, 1297, 2346, 3424, 9389, 3717, 79653, 3969, 1078, 211400, 205408, 644, 21360, 13, 71056, 1649, 19450, 94187, 176761, 324, 90032, 166568, 11, 25606, 1725, 2972, 1323, 43391, 5346, 194456, 332, 1740, 324, 84725, 11797, 7657, 204928, 1725, 13958, 5346, 46213, 1649, 62793, 3656, 1147, 21360, 13, 1526, 5346, 11, 5992, 46207, 401, 1147, 184743, 1725, 156171, 7639, 7657, 69073, 1778, 9750, 43503, 8098, 11, 3633, 7528, 23889, 2093, 846, 909, 240219, 160867, 5724, 909, 227128, 4259, 1147, 184743, 173275, 318, 32784, 478, 13881, 8, 1725, 220168, 284, 4783, 13, 248046]
```

## explanation.r1.candidate

Commit moteur : `cdd426e1670a95c1798e45962c48e8d7ae87daaf`. Arrêt : EOS.

### Prompt intégral

```text
Explique en français, en 100 mots maximum, comment un modèle d'intelligence artificielle de 24 Go peut fonctionner sur un ordinateur possédant seulement 16 Go de mémoire vive.
```

### Réponse affichée intégrale

```text
Un modèle de 24 Go peut fonctionner sur 16 Go de RAM grâce au **swap** (mémoire virtuelle). Le système d’exploitation déplace les données inactives du modèle depuis la RAM vers le disque dur (SSD/HDD), libérant ainsi de l’espace mémoire. Seules les parties du modèle activement utilisées restent en RAM. Bien que cette méthode permette l’exécution, elle est **beaucoup plus lente** car l’accès au disque est bien plus lent que celui à la RAM. De plus, une partie de la mémoire est réservée au système et aux autres applications, ce qui peut causer des ralentissements ou des erreurs si la mémoire totale (RAM + swap) est insuffisante.
```

### Sortie brute intégrale

```text
Un modèle de 24 Go peut fonctionner sur 16 Go de RAM grâce au **swap** (mémoire virtuelle). Le système d’exploitation déplace les données inactives du modèle depuis la RAM vers le disque dur (SSD/HDD), libérant ainsi de l’espace mémoire. Seules les parties du modèle activement utilisées restent en RAM. Bien que cette méthode permette l’exécution, elle est **beaucoup plus lente** car l’accès au disque est bien plus lent que celui à la RAM. De plus, une partie de la mémoire est réservée au système et aux autres applications, ce qui peut causer des ralentissements ou des erreurs si la mémoire totale (RAM + swap) est insuffisante.<|im_end|>
```

### IDs générés, EOS inclus

```json
[1743, 79653, 401, 220, 17, 19, 5805, 23889, 237805, 1668, 220, 16, 21, 5805, 401, 21360, 73721, 7657, 2972, 24921, 332, 318, 155769, 93064, 224456, 553, 1900, 69073, 293, 515, 193400, 7204, 1940, 3424, 44064, 303, 519, 1821, 3717, 79653, 38767, 1147, 21360, 5266, 501, 204928, 10336, 318, 1178, 35, 22629, 3967, 681, 2953, 185536, 41825, 401, 324, 86056, 1260, 184743, 13, 1297, 2346, 3424, 9389, 3717, 79653, 3969, 1078, 211400, 205408, 644, 21360, 13, 71056, 1649, 19450, 94187, 176761, 324, 90032, 166568, 11, 25606, 1725, 2972, 1323, 43391, 5346, 194456, 332, 1740, 324, 84725, 11797, 7657, 204928, 1725, 13958, 5346, 46213, 1649, 62793, 3656, 1147, 21360, 13, 1526, 5346, 11, 5992, 46207, 401, 1147, 184743, 1725, 156171, 7639, 7657, 69073, 1778, 9750, 43503, 8098, 11, 3633, 7528, 23889, 2093, 846, 909, 240219, 160867, 5724, 909, 227128, 4259, 1147, 184743, 173275, 318, 32784, 478, 13881, 8, 1725, 220168, 284, 4783, 13, 248046]
```

## long_context.r1.candidate

Commit moteur : `cdd426e1670a95c1798e45962c48e8d7ae87daaf`. Arrêt : EOS.

### Prompt intégral

```text
Dossier technique fictif pour une expérience locale. L'ordinateur possède 16 Gio de mémoire unifiée et utilise son SSD interne. Le modèle est stocké dans un fichier GGUF de 23,94 Go. Le système peut lire sélectivement les poids nécessaires à chaque couche. Le cache de poids est limité à 64 Mio, le budget des entrées et sorties à 64 Mio, les états de génération à 256 Mio et l'espace de travail à 128 Mio. La réserve système de 3 Gio est indépendante de ces budgets. Le superviseur vérifie la pression mémoire et l'augmentation du swap toutes les 50 millisecondes. Aucune application personnelle n'est fermée. Le modèle conserve ses poids d'origine. Question : quelle est la somme des quatre budgets de cache, entrées et sorties, états et travail ? Réponds par une seule phrase, en Mio, sans inclure la réserve système.
```

### Réponse affichée intégrale

```text
La somme des quatre budgets de cache, entrées et sorties, états et travail est de 512 Mio.
```

### Sortie brute intégrale

```text
La somme des quatre budgets de cache, entrées et sorties, états et travail est de 512 Mio.<|im_end|>
```

### IDs générés, EOS inclus

```json
[8482, 162622, 909, 94821, 40016, 401, 6297, 11, 11161, 13306, 1778, 218800, 11, 13516, 1798, 1778, 40298, 1725, 401, 220, 20, 16, 17, 187631, 13, 248046]
```

## long_context.r1.baseline

Commit moteur : `ab59677fed6181d909866364a50dd07244af6b1b`. Arrêt : EOS.

### Prompt intégral

```text
Dossier technique fictif pour une expérience locale. L'ordinateur possède 16 Gio de mémoire unifiée et utilise son SSD interne. Le modèle est stocké dans un fichier GGUF de 23,94 Go. Le système peut lire sélectivement les poids nécessaires à chaque couche. Le cache de poids est limité à 64 Mio, le budget des entrées et sorties à 64 Mio, les états de génération à 256 Mio et l'espace de travail à 128 Mio. La réserve système de 3 Gio est indépendante de ces budgets. Le superviseur vérifie la pression mémoire et l'augmentation du swap toutes les 50 millisecondes. Aucune application personnelle n'est fermée. Le modèle conserve ses poids d'origine. Question : quelle est la somme des quatre budgets de cache, entrées et sorties, états et travail ? Réponds par une seule phrase, en Mio, sans inclure la réserve système.
```

### Réponse affichée intégrale

```text
La somme des quatre budgets de cache, entrées et sorties, états et travail est de 512 Mio.
```

### Sortie brute intégrale

```text
La somme des quatre budgets de cache, entrées et sorties, états et travail est de 512 Mio.<|im_end|>
```

### IDs générés, EOS inclus

```json
[8482, 162622, 909, 94821, 40016, 401, 6297, 11, 11161, 13306, 1778, 218800, 11, 13516, 1798, 1778, 40298, 1725, 401, 220, 20, 16, 17, 187631, 13, 248046]
```

## short.r0.baseline

Commit moteur : `ab59677fed6181d909866364a50dd07244af6b1b`. Arrêt : EOS.

### Prompt intégral

```text
Cite les trois couleurs du drapeau français, sans autre explication.
```

### Réponse affichée intégrale

```text
Bleu, blanc, rouge
```

### Sortie brute intégrale

```text
Bleu, blanc, rouge<|im_end|>
```

### IDs générés, EOS inclus

```json
[33, 273, 84, 11, 61112, 11, 81063, 248046]
```

## short.r0.candidate

Commit moteur : `cdd426e1670a95c1798e45962c48e8d7ae87daaf`. Arrêt : EOS.

### Prompt intégral

```text
Cite les trois couleurs du drapeau français, sans autre explication.
```

### Réponse affichée intégrale

```text
Bleu, blanc, rouge
```

### Sortie brute intégrale

```text
Bleu, blanc, rouge<|im_end|>
```

### IDs générés, EOS inclus

```json
[33, 273, 84, 11, 61112, 11, 81063, 248046]
```

## short.r1.candidate

Commit moteur : `cdd426e1670a95c1798e45962c48e8d7ae87daaf`. Arrêt : EOS.

### Prompt intégral

```text
Cite les trois couleurs du drapeau français, sans autre explication.
```

### Réponse affichée intégrale

```text
Bleu, blanc, rouge
```

### Sortie brute intégrale

```text
Bleu, blanc, rouge<|im_end|>
```

### IDs générés, EOS inclus

```json
[33, 273, 84, 11, 61112, 11, 81063, 248046]
```

## short.r1.baseline

Commit moteur : `ab59677fed6181d909866364a50dd07244af6b1b`. Arrêt : EOS.

### Prompt intégral

```text
Cite les trois couleurs du drapeau français, sans autre explication.
```

### Réponse affichée intégrale

```text
Bleu, blanc, rouge
```

### Sortie brute intégrale

```text
Bleu, blanc, rouge<|im_end|>
```

### IDs générés, EOS inclus

```json
[33, 273, 84, 11, 61112, 11, 81063, 248046]
```

## long_context.r0.baseline

Commit moteur : `ab59677fed6181d909866364a50dd07244af6b1b`. Arrêt : EOS.

### Prompt intégral

```text
Dossier technique fictif pour une expérience locale. L'ordinateur possède 16 Gio de mémoire unifiée et utilise son SSD interne. Le modèle est stocké dans un fichier GGUF de 23,94 Go. Le système peut lire sélectivement les poids nécessaires à chaque couche. Le cache de poids est limité à 64 Mio, le budget des entrées et sorties à 64 Mio, les états de génération à 256 Mio et l'espace de travail à 128 Mio. La réserve système de 3 Gio est indépendante de ces budgets. Le superviseur vérifie la pression mémoire et l'augmentation du swap toutes les 50 millisecondes. Aucune application personnelle n'est fermée. Le modèle conserve ses poids d'origine. Question : quelle est la somme des quatre budgets de cache, entrées et sorties, états et travail ? Réponds par une seule phrase, en Mio, sans inclure la réserve système.
```

### Réponse affichée intégrale

```text
La somme des quatre budgets de cache, entrées et sorties, états et travail est de 512 Mio.
```

### Sortie brute intégrale

```text
La somme des quatre budgets de cache, entrées et sorties, états et travail est de 512 Mio.<|im_end|>
```

### IDs générés, EOS inclus

```json
[8482, 162622, 909, 94821, 40016, 401, 6297, 11, 11161, 13306, 1778, 218800, 11, 13516, 1798, 1778, 40298, 1725, 401, 220, 20, 16, 17, 187631, 13, 248046]
```

## long_context.r0.candidate

Commit moteur : `cdd426e1670a95c1798e45962c48e8d7ae87daaf`. Arrêt : EOS.

### Prompt intégral

```text
Dossier technique fictif pour une expérience locale. L'ordinateur possède 16 Gio de mémoire unifiée et utilise son SSD interne. Le modèle est stocké dans un fichier GGUF de 23,94 Go. Le système peut lire sélectivement les poids nécessaires à chaque couche. Le cache de poids est limité à 64 Mio, le budget des entrées et sorties à 64 Mio, les états de génération à 256 Mio et l'espace de travail à 128 Mio. La réserve système de 3 Gio est indépendante de ces budgets. Le superviseur vérifie la pression mémoire et l'augmentation du swap toutes les 50 millisecondes. Aucune application personnelle n'est fermée. Le modèle conserve ses poids d'origine. Question : quelle est la somme des quatre budgets de cache, entrées et sorties, états et travail ? Réponds par une seule phrase, en Mio, sans inclure la réserve système.
```

### Réponse affichée intégrale

```text
La somme des quatre budgets de cache, entrées et sorties, états et travail est de 512 Mio.
```

### Sortie brute intégrale

```text
La somme des quatre budgets de cache, entrées et sorties, états et travail est de 512 Mio.<|im_end|>
```

### IDs générés, EOS inclus

```json
[8482, 162622, 909, 94821, 40016, 401, 6297, 11, 11161, 13306, 1778, 218800, 11, 13516, 1798, 1778, 40298, 1725, 401, 220, 20, 16, 17, 187631, 13, 248046]
```
