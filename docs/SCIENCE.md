# Base LCM — Fondements scientifiques

## 1. Motivation : au-delà de la modélisation token à token

Les grands modèles de langage (LLMs) actuels opèrent à l'échelle du token : ils prédisent un sous-mot après l'autre, ce qui les oblige à consacrer une grande partie de leur capacité paramétrique à des dépendances de surface (orthographe, ponctuation, syntaxe locale). Cette granularité fine est à la fois une force — elle permet une génération fluide — et une limite : raisonner sur de longues dépendances sémantiques requiert de propager de l'information sur des centaines de tokens.

L'hypothèse de ce projet est inspirée du papier **Large Concept Models** (Favre et al., Meta FAIR, 2024) : il est possible d'entraîner un modèle autorégressif *directement dans un espace de représentations sémantiques continues* (les "concepts"), en laissant un encodeur-décodeur séparé gérer la conversion texte ↔ concept. Le modèle de concepts se concentre alors sur la structure discursive de haut niveau, indépendamment de la langue ou de la surface textuelle.

---

## 2. L'espace de concepts : embeddings SONAR

### Architecture SONAR

SONAR (Sentence-level multimOdal and laNguage-Agnostic Representations, Duquenne et al., Meta FAIR, 2023) est un espace d'embedding de phrases entraîné sur des données parallèles multilingues avec un objectif de reconstruction (auto-encodeur au niveau phrase). Il produit des vecteurs denses de dimension **1024** qui satisfont plusieurs propriétés utiles :

- **Invariance linguistique** : deux phrases de sens proche dans des langues différentes ont des embeddings proches en cosinus.
- **Sémantique compositionnelle** : les vecteurs capturent le sens de la phrase entière, pas seulement des mots-clés.
- **Décodabilité** : un décodeur SONAR peut reconstruire du texte naturel depuis un vecteur, ce qui permet d'évaluer la qualité des prédictions par reconstruction.

### Pourquoi SONAR est un bon espace de concepts

Un "concept" au sens LCM est une unité de sens autonome — typiquement une phrase ou un paragraphe. SONAR est l'espace le plus adapté car :

1. Il est **continu et lisse** : l'interpolation entre deux embeddings produit des représentations intermédiaires cohérentes.
2. Il est **déjà préentraîné sur 200+ langues** : le modèle de concepts hérite de la multilingualité sans supervision supplémentaire.
3. Il a une **structure géométrique exploitable** : la distance cosinus reflète la similarité sémantique, ce qui donne un signal de loss intrinsèquement significatif.

---

## 3. Objectif d'entraînement : prédiction du prochain concept

### Formulation

Étant donné une séquence de concepts $\mathbf{c}_1, \mathbf{c}_2, \ldots, \mathbf{c}_T$ (embeddings SONAR de phrases successives d'un document), le modèle apprend la distribution conditionnelle :

$$p(\mathbf{c}_{t+1} \mid \mathbf{c}_1, \ldots, \mathbf{c}_t)$$

Contrairement à la modélisation de langage classique sur tokens discrets (softmax sur un vocabulaire), ici la cible est un vecteur continu de $\mathbb{R}^{1024}$. La loss est donc une **RMSE dans l'espace d'embedding** :

$$\mathcal{L} = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \left\| \hat{\mathbf{c}}_{t+1} - \mathbf{c}_{t+1} \right\|_2$$

où $\mathcal{T}$ est l'ensemble des positions non-paddées et $\hat{\mathbf{c}}_{t+1}$ est la prédiction du modèle.

### Analogie et différence avec la modélisation de langage

| | LLM token-à-token | Base LCM (ce projet) |
|---|---|---|
| Unité de base | Sous-mot (~3 chars) | Phrase entière |
| Espace de sortie | Discret (vocabulaire ~32k) | Continu ($\mathbb{R}^{1024}$) |
| Loss | Cross-entropie | RMSE |
| Inférence | Sampling/greedy sur logits | Prédiction directe du vecteur |
| Granularité temporelle | ~10 tokens/phrase | 1 concept/phrase |

### Propriété causale (masque triangulaire)

Pour que la prédiction de $\mathbf{c}_{t+1}$ ne puisse pas "voir" $\mathbf{c}_{t+1}, \ldots, \mathbf{c}_T$ pendant l'entraînement, l'attention du Transformer est masquée causalement : la position $i$ ne peut attendre que les positions $j \leq i$. Sans ce masque, le modèle apprend trivialement à copier le concept voisin (fuite d'information), mais s'effondre en inférence où ce raccourci n'existe plus.

---

## 4. Architecture : Transformer décodeur en espace d'embedding

### Vue d'ensemble

```
SONAR embeddings (B, S, 1024)
        │
    ┌───▼───────────────┐
    │      PreNet        │  normalisation + projection 1024 → 2048
    │  StandardScaler    │  + encodage positionnel sinusoïdal
    │  Linear(1024→2048) │
    │  SinusoidalPE      │
    └───┬───────────────┘
        │ (B, S, 2048)
    ┌───▼───────────────────────────────────┐
    │   Transformer Décodeur (12 couches)    │
    │  ┌─────────────────────────────────┐  │
    │  │  Pre-LayerNorm (DyT)            │  │
    │  │  QK-Normed Multihead Attention  │  │
    │  │  + masque causal + padding mask │  │
    │  │  Residual + Dropout             │  │
    │  │  Pre-LayerNorm (DyT)            │  │
    │  │  FFN (Linear → GELU → Linear)  │  │
    │  │  Residual + Dropout             │  │
    │  └─────────────────────────────────┘  │
    └───┬───────────────────────────────────┘
        │ (B, S, 2048)
    ┌───▼───────────────┐
    │     PostNet        │  déprojection 2048 → 1024
    │  Linear(2048→1024) │  + dénormalisation (inverse de PreNet)
    └───┬───────────────┘
        │
SONAR embeddings prédits (B, S, 1024)
```

### Composants clés

**StandardScaler adaptatif** : normalise les embeddings d'entrée avec une moyenne exponentielle des statistiques observées (momentum décroissant de $0.1 \cdot e^{-n/3000}$). En évaluation, utilise les statistiques figées accumulées pendant l'entraînement. La PostNet applique l'inverse de cette normalisation pour restituer des vecteurs dans l'espace SONAR original.

**Encodage positionnel sinusoïdal** : vecteurs de position de Vaswani et al. (2017) appliqués dans l'espace caché après projection :

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**QK-Normed Attention** : les projections Query et Key sont suivies d'une LayerNorm avant le calcul des scores d'attention, ce qui stabilise l'entraînement en évitant les gradients explosifs dans les logits d'attention (technique issue de Dehghani et al., 2023).

**DyT (Dynamic Tanh)** : alternative à la LayerNorm qui remplace la normalisation statistique par $\gamma \cdot \tanh(\alpha x) + \beta$ avec $\alpha, \gamma, \beta$ appris. Élimine le calcul de moyenne/variance par batch et offre une normalisation bornée naturellement.

---

## 5. Données : Wikipedia multilingue

### Pourquoi Wikipedia

Wikipedia est un corpus de choix pour le préentraînement LCM car :

1. **Structure discursive naturelle** : les articles sont organisés en sections et paragraphes qui forment des séquences de concepts cohérents — exactement ce que le modèle doit apprendre.
2. **Diversité thématique** : sciences, histoire, géographie, culture — le modèle est exposé à une grande variété de structures argumentatives.
3. **Qualité éditoriale** : les phrases de Wikipedia sont généralement bien formées et informationnellement denses, ce qui produit des embeddings SONAR de haute qualité.
4. **Multilingualité** : Wikipedia existe en 300+ langues. En combinant plusieurs langues, le modèle LCM hérite de la robustesse multilingue de SONAR.

### Pipeline de prétraitement

```
Article Wikipedia (texte brut)
    │
    ▼  Segmentation en phrases (spaCy)
Phrases [s_1, s_2, ..., s_N]
    │
    ▼  Encodage SONAR
Embeddings [c_1, c_2, ..., c_N] ∈ ℝ^{N×1024}
    │
    ▼  Sauvegarde par document (FileSystemEmbeddingRepository)
.pt fichiers indexés par document_id
    │
    ▼  EmbeddingsDataset (fenêtre glissante, stride=16)
Séquences (input[:-1], target[1:], padding_mask)
```

---

## 6. Métriques d'évaluation

### Loss d'entraînement (RMSE masquée)

La loss principale est la RMSE sur les positions non-paddées, scalée par 100 pour lisibilité :

$$\mathcal{L}_{\text{train}} = 100 \cdot \sqrt{\frac{\sum_{t} m_t \cdot \|\hat{\mathbf{c}}_t - \mathbf{c}_t\|_2^2}{\sum_t m_t} + \varepsilon}$$

où $m_t \in \{0,1\}$ est le padding mask.

### Similarité cosinus (évaluation qualitative)

Pour évaluer si les prédictions restent dans la bonne région de l'espace SONAR :

$$\text{cos\_sim}(\hat{\mathbf{c}}, \mathbf{c}) = \frac{\hat{\mathbf{c}} \cdot \mathbf{c}}{\|\hat{\mathbf{c}}\| \cdot \|\mathbf{c}\|}$$

Un score > 0.7 indique que le concept prédit est sémantiquement proche du concept cible.

### Reconstruction textuelle (évaluation qualitative)

Le décodeur SONAR permet de reconvertir les embeddings prédits en texte, offrant une évaluation humaine directe :

$$\hat{\mathbf{c}}_{t+1} \xrightarrow{\text{SONAR decoder}} \text{texte généré}$$

Cette évaluation détecte des défauts invisibles dans l'espace d'embedding (effondrement vers un point fixe, répétition, dérive hors distribution).

---

## 7. Limites et travaux futurs

### Limites actuelles

- **PostNet simple** : une seule couche linéaire de projection. Le papier LCM original utilise un réseau plus profond (Linear → GELU → Dropout → LayerNorm → Linear) pour une meilleure reconstruction.
- **Sinusoïdal PE** : l'encodage positionnel actuel est additif et partagé entre Q, K, V. Le **vrai RoPE** (Rotary Position Embedding, Su et al., 2021) applique des rotations position-dépendantes uniquement sur Q et K à l'intérieur de l'attention, ce qui améliore la généralisation aux longueurs de séquence non vues à l'entraînement.
- **Padding inefficace** : les séquences sont paddées à longueur fixe. Le **sequence packing** (concaténation de plusieurs documents courts avec masque causal par segment) éliminerait le padding et augmenterait le throughput de 20-40%.

### Directions futures

| Direction | Gain attendu | Complexité |
|---|---|---|
| Vrai RoPE dans QKNormedMultiheadAttention | Meilleure généralisation longueur | Moyenne |
| PostNet profond (Linear→GELU→LN→Linear) | Meilleure reconstruction | Faible |
| Sequence packing + block-diagonal causal mask | +30% throughput, 0% padding | Moyenne |
| SemDeDup (déduplication sémantique FAISS) | -30% données, +diversité | Faible |
| Stockage memmap bf16 | -50% I/O disque | Faible |
| Post-training sur données d'instruction | Following d'instructions au niveau concept | Haute |

---

## Références

- Favre et al. (2024). *Large Concept Models: Language Modeling in a Sentence Representation Space*. Meta FAIR.
- Duquenne et al. (2023). *SONAR: Sentence-Level Multimodal and Language-Agnostic Representations*. Meta FAIR.
- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864.
- Dehghani et al. (2023). *Scaling Vision Transformers to 22 Billion Parameters*. ICML. (QK-Norm)
- Zhu & Kiros (2025). *Transformers without Normalization* (Dynamic Tanh). arXiv:2503.10622.
