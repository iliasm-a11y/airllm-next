# Réponses intégrales — campagne HyperStream

Campagne complète et appariée. Tokens, logits, routes et positions concordent par paire. Les textes ci-dessous sont les sorties enregistrées, sans réécriture. Le token EOS figure dans les IDs et dans le texte brut.

## short.r0.baseline

Commit moteur : `ca3a42935be876bbdbbc1e2adbd16c8547746175`. Arrêt : EOS.

### Prompt intégral

```text
J'ai 180 euros. Je dépense 65 euros puis 91 euros. Combien reste-t-il ? Réponds par une seule phrase courte.
```

### Réponse affichée intégrale

```text
Il reste 24 euros.
```

### Sortie brute intégrale

```text
Il reste 24 euros.<|im_end|>
```

### IDs générés, EOS inclus

```json
[11707, 62869, 220, 17, 19, 31415, 13, 248046]
```

## short.r0.candidate

Commit moteur : `6960ef97ac4aa71d191087320362c1d636562aed`. Arrêt : EOS.

### Prompt intégral

```text
J'ai 180 euros. Je dépense 65 euros puis 91 euros. Combien reste-t-il ? Réponds par une seule phrase courte.
```

### Réponse affichée intégrale

```text
Il reste 24 euros.
```

### Sortie brute intégrale

```text
Il reste 24 euros.<|im_end|>
```

### IDs générés, EOS inclus

```json
[11707, 62869, 220, 17, 19, 31415, 13, 248046]
```

## explanation.r0.candidate

Commit moteur : `6960ef97ac4aa71d191087320362c1d636562aed`. Arrêt : EOS.

### Prompt intégral

```text
Explique en deux phrases courtes pourquoi lire les poids d'un modèle depuis un SSD peut limiter sa vitesse, même avec un GPU rapide.
```

### Réponse affichée intégrale

```text
Le SSD, bien que rapide, reste un goulot d'étranglement par rapport à la bande passante interne du GPU, limitant le débit de données. Cette latence d'entrée/sortie empêche le processeur graphique d'être alimenté en continu, réduisant ainsi son utilisation effective.
```

### Sortie brute intégrale

```text
Le SSD, bien que rapide, reste un goulot d'étranglement par rapport à la bande passante interne du GPU, limitant le débit de données. Cette latence d'entrée/sortie empêche le processeur graphique d'être alimenté en continu, réduisant ainsi son utilisation effective.<|im_end|>
```

### IDs générés, EOS inclus

```json
[2229, 35160, 11, 13958, 1649, 72931, 11, 62869, 632, 88861, 8911, 293, 29668, 221554, 954, 1298, 37102, 3656, 1147, 90793, 1440, 4783, 177470, 3717, 21966, 11, 3798, 505, 501, 221018, 401, 44064, 13, 59238, 6630, 738, 293, 6, 176996, 2600, 366, 627, 8226, 162447, 501, 27375, 323, 322, 211216, 293, 89529, 36115, 933, 644, 15647, 11, 157521, 177956, 41825, 4292, 184843, 7139, 13, 248046]
```

## explanation.r0.baseline

Commit moteur : `ca3a42935be876bbdbbc1e2adbd16c8547746175`. Arrêt : EOS.

### Prompt intégral

```text
Explique en deux phrases courtes pourquoi lire les poids d'un modèle depuis un SSD peut limiter sa vitesse, même avec un GPU rapide.
```

### Réponse affichée intégrale

```text
Le SSD, bien que rapide, reste un goulot d'étranglement par rapport à la bande passante interne du GPU, limitant le débit de données. Cette latence d'entrée/sortie empêche le processeur graphique d'être alimenté en continu, réduisant ainsi son utilisation effective.
```

### Sortie brute intégrale

```text
Le SSD, bien que rapide, reste un goulot d'étranglement par rapport à la bande passante interne du GPU, limitant le débit de données. Cette latence d'entrée/sortie empêche le processeur graphique d'être alimenté en continu, réduisant ainsi son utilisation effective.<|im_end|>
```

### IDs générés, EOS inclus

```json
[2229, 35160, 11, 13958, 1649, 72931, 11, 62869, 632, 88861, 8911, 293, 29668, 221554, 954, 1298, 37102, 3656, 1147, 90793, 1440, 4783, 177470, 3717, 21966, 11, 3798, 505, 501, 221018, 401, 44064, 13, 59238, 6630, 738, 293, 6, 176996, 2600, 366, 627, 8226, 162447, 501, 27375, 323, 322, 211216, 293, 89529, 36115, 933, 644, 15647, 11, 157521, 177956, 41825, 4292, 184843, 7139, 13, 248046]
```

## short.r1.candidate

Commit moteur : `6960ef97ac4aa71d191087320362c1d636562aed`. Arrêt : EOS.

### Prompt intégral

```text
J'ai 180 euros. Je dépense 65 euros puis 91 euros. Combien reste-t-il ? Réponds par une seule phrase courte.
```

### Réponse affichée intégrale

```text
Il reste 24 euros.
```

### Sortie brute intégrale

```text
Il reste 24 euros.<|im_end|>
```

### IDs générés, EOS inclus

```json
[11707, 62869, 220, 17, 19, 31415, 13, 248046]
```

## short.r1.baseline

Commit moteur : `ca3a42935be876bbdbbc1e2adbd16c8547746175`. Arrêt : EOS.

### Prompt intégral

```text
J'ai 180 euros. Je dépense 65 euros puis 91 euros. Combien reste-t-il ? Réponds par une seule phrase courte.
```

### Réponse affichée intégrale

```text
Il reste 24 euros.
```

### Sortie brute intégrale

```text
Il reste 24 euros.<|im_end|>
```

### IDs générés, EOS inclus

```json
[11707, 62869, 220, 17, 19, 31415, 13, 248046]
```

## explanation.r1.baseline

Commit moteur : `ca3a42935be876bbdbbc1e2adbd16c8547746175`. Arrêt : EOS.

### Prompt intégral

```text
Explique en deux phrases courtes pourquoi lire les poids d'un modèle depuis un SSD peut limiter sa vitesse, même avec un GPU rapide.
```

### Réponse affichée intégrale

```text
Le SSD, bien que rapide, reste un goulot d'étranglement par rapport à la bande passante interne du GPU, limitant le débit de données. Cette latence d'entrée/sortie empêche le processeur graphique d'être alimenté en continu, réduisant ainsi son utilisation effective.
```

### Sortie brute intégrale

```text
Le SSD, bien que rapide, reste un goulot d'étranglement par rapport à la bande passante interne du GPU, limitant le débit de données. Cette latence d'entrée/sortie empêche le processeur graphique d'être alimenté en continu, réduisant ainsi son utilisation effective.<|im_end|>
```

### IDs générés, EOS inclus

```json
[2229, 35160, 11, 13958, 1649, 72931, 11, 62869, 632, 88861, 8911, 293, 29668, 221554, 954, 1298, 37102, 3656, 1147, 90793, 1440, 4783, 177470, 3717, 21966, 11, 3798, 505, 501, 221018, 401, 44064, 13, 59238, 6630, 738, 293, 6, 176996, 2600, 366, 627, 8226, 162447, 501, 27375, 323, 322, 211216, 293, 89529, 36115, 933, 644, 15647, 11, 157521, 177956, 41825, 4292, 184843, 7139, 13, 248046]
```

## explanation.r1.candidate

Commit moteur : `6960ef97ac4aa71d191087320362c1d636562aed`. Arrêt : EOS.

### Prompt intégral

```text
Explique en deux phrases courtes pourquoi lire les poids d'un modèle depuis un SSD peut limiter sa vitesse, même avec un GPU rapide.
```

### Réponse affichée intégrale

```text
Le SSD, bien que rapide, reste un goulot d'étranglement par rapport à la bande passante interne du GPU, limitant le débit de données. Cette latence d'entrée/sortie empêche le processeur graphique d'être alimenté en continu, réduisant ainsi son utilisation effective.
```

### Sortie brute intégrale

```text
Le SSD, bien que rapide, reste un goulot d'étranglement par rapport à la bande passante interne du GPU, limitant le débit de données. Cette latence d'entrée/sortie empêche le processeur graphique d'être alimenté en continu, réduisant ainsi son utilisation effective.<|im_end|>
```

### IDs générés, EOS inclus

```json
[2229, 35160, 11, 13958, 1649, 72931, 11, 62869, 632, 88861, 8911, 293, 29668, 221554, 954, 1298, 37102, 3656, 1147, 90793, 1440, 4783, 177470, 3717, 21966, 11, 3798, 505, 501, 221018, 401, 44064, 13, 59238, 6630, 738, 293, 6, 176996, 2600, 366, 627, 8226, 162447, 501, 27375, 323, 322, 211216, 293, 89529, 36115, 933, 644, 15647, 11, 157521, 177956, 41825, 4292, 184843, 7139, 13, 248046]
```

