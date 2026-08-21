# Stratégie d'entraînement — Base LCM

Le paradigme pré-entraînement / fine-tuning / post-entraînement est celui qui a fait le succès des LLMs modernes. Il s'applique naturellement aux LCMs, avec la particularité que toutes les phases opèrent dans l'espace SONAR plutôt qu'en tokens.

---

## Phase 1 — Pré-entraînement

**Objectif** : apprendre la structure du monde en espace de concepts.

Le modèle voit des paires `(séquence_entrée, concept_suivant)` et apprend une représentation interne de *"comment les idées s'enchaînent"*. C'est la phase la plus coûteuse en compute et en données, mais elle n'a besoin d'aucun label humain — Wikipedia suffit.

À la fin, le modèle sait que *"concept sur les Alpes" → "concept sur le ski"* est probable, mais il ne sait pas *quoi faire* avec cette capacité.

**Données** : Wikipedia multilingue (embeddings SONAR préalablement calculés)  
**Loss** : RMSE masquée entre le concept prédit et le concept suivant réel  
**Script** : `scripts/train_base_lcm.py`

---

## Phase 2 — Fine-tuning supervisé (SFT)

**Objectif** : apprendre un comportement utile.

On prend le modèle pré-entraîné et on le ré-entraîne sur des **séquences structurées** : paires `(résumé, développement)`, `(question, réponse argumentée)`, ou séquences narratives — toujours en espace SONAR, pas en tokens.

Le modèle apprend à *générer* des séquences de concepts cohérentes avec une intention, pas juste à prédire le suivant statistiquement.

**Données candidates** :
- `LogiCoT` — enchaînements causaux et logiques (`scripts/process_logicot.py`)
- `ROC Stories` — séquences narratives courtes et cohérentes (`scripts/process_roc_stories.py`)

Ces deux datasets sont déjà intégrés dans le repo et constituent un bon point de départ pour valider que le modèle peut produire des enchaînements intentionnels dans un domaine contrôlé.

---

## Phase 3 — Post-entraînement (DPO / RLHF)

**Objectif** : aligner le modèle sur des préférences humaines ou une métrique de qualité.

On entraîne le modèle à produire des séquences de concepts que des humains (ou un modèle de récompense) préfèrent :

- **DPO** (Direct Preference Optimization) : paires `(bonne séquence, mauvaise séquence)` annotées — plus simple à mettre en œuvre
- **RLHF** : un modèle de récompense qui score les séquences générées en continu

Dans l'espace SONAR, la récompense peut être mesurée directement en similarité cosinus avec un concept cible, sans avoir besoin de décoder en texte — ce qui simplifie considérablement la boucle de renforcement.

---

## Vue d'ensemble

```
Wikipedia (SONAR)          LogiCoT / ROC Stories       Préférences humaines
       │                           │                           │
       ▼                           ▼                           ▼
  Pré-entraînement           Fine-tuning SFT            Post-entraînement
  next-concept pred.    séquences intentionnelles      DPO / RLHF concept

  "comment les idées      "comment résoudre /           "quelle séquence
   s'enchaînent"          raconter quelque chose"        est la meilleure"
```

---

## Recommandation pratique

Le pré-entraînement est la bonne première étape — sans lui, les deux phases suivantes ne peuvent rien apprendre de profond.

Pour valider que l'architecture fonctionne *avant* d'investir dans un long pré-entraînement :

1. **Quelques milliers de steps sur Wikipedia** — vérifier que la loss descend et que la génération autorégrressive diverge (ne converge plus vers un point fixe grâce au masque causal)
2. **Fine-tuning rapide sur ROC Stories** — séquences courtes et narratives, idéales pour vérifier que le modèle produit des concepts cohérents dans un domaine contrôlé
3. **Post-entraînement** en dernier, quand le SFT montre déjà un comportement raisonnable
