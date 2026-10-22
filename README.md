# AirLLM Next

Livraison complète du 22 octobre 2026, issue du véritable AirLLM et de la version Ultra fournie.

**[Télécharger le projet complet — AirLLM-Next-2026-10-22.zip](AirLLM-Next-2026-10-22.zip)**

L’archive contient les sources modifiées, le backend natif MLX GLM-MoE-DSA, le streaming sélectif, le cache d’experts, les conversions de quantification, les scripts macOS, les tests, les benchmarks, les journaux et les rapports de validation.

Les sources complètes sont dans le ZIP. Le [guide détaillé](README_NEXT.md) est également consultable directement dans ce dépôt.

## Installation sur le Mac

Décompresser le ZIP, ouvrir un terminal dans son dossier `airllm-next`, puis exécuter :

```bash
bash scripts/setup_next_macos.sh
bash scripts/validate_next_macos.sh
```

Le script vérifie l’environnement avant l’installation. Aucun poids de modèle n’est téléchargé automatiquement.

Une fois les poids locaux présents :

```bash
bash scripts/launch_next_macos.sh /Volumes/SSD/GLM-5.3 --prompt "Bonjour" --max-new-tokens 16
```

## Validation et limites

La livraison a été testée sur CPU Linux avec MLX : **135 tests réussis, 9 tests matériels ignorés et 5 sous-tests réussis**. Les détails, les commandes et les résultats sont inclus dans l’archive.

**Metal, Apple M5, le SSD Fanxiang et le checkpoint complet de GLM-5.3 restent à valider sur le matériel réel.** Aucun benchmark M5 ni débit de génération du modèle complet n’est annoncé.

## Intégrité

SHA-256 du ZIP :

```text
7433255c37830b6d1603e276452154a81e67c7bdbd6eae826216230b1bb93e2d
```

Les licences et attributions d’origine sont conservées dans l’archive et dans les fichiers `LICENSE`, `NOTICE_NEXT.txt` et `NOTICE_ULTRA.txt` de ce dépôt.
