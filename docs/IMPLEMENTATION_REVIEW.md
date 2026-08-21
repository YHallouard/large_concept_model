# Revue d'implémentation — SONAR / LCM / Data

*Audit réalisé par rapport aux papiers de référence :*
- *SONAR* — [arxiv 2208.12218](https://arxiv.org/abs/2208.12218)
- *LCM* — [arxiv 2412.08821](https://arxiv.org/abs/2412.08821)

---

## 1. Résumé exécutif

Trois problèmes critiques ont été identifiés et corrigés dans ce cycle de refonte :

| Priorité | Zone | Problème | Impact |
|---|---|---|---|
| 🔴 | SONAR (encodeur) | Encodage phrase par phrase sans masked mean-pool | Erreur de représentation à chaque batch padded |
| 🔴 | Base-LCM | Normalisation scalaire EMA au lieu de stats par-dimension figées | Gradient faussé — dimensions à grande échelle dominent |
| 🔴 | One-Tower LCM | Bruit additif non-variance-preserving + self-attention sans masque causal | Modèle de diffusion non fonctionnel + fuite d'information future |

Les correctifs sont décrits section par section ci-dessous, accompagnés de l'état après refonte.

---

## 2. SONAR

### 2.1 Rappel du papier

SONAR (Duquenne et al., Meta FAIR, 2023) est un encodeur de phrases multilingues basé sur une architecture NLLB/M2M-100. L'embedding de phrase est obtenu par **masked mean-pool** sur les états cachés du dernier Transformer encoder :

```
emb = sum(h_i * mask_i) / sum(mask_i)
```

Les tokens de padding doivent être **exclus** du calcul (d'où le masque). L'espace résultant est de dimension **1024**.

### 2.2 Constats

| # | Sév. | Constat | Fichier | Statut |
|---|---|---|---|---|
| S1 | 🟠 | `spacy.load(...)` appelé **à l'import** du module → crash collecte pytest et tout import sans le modèle installé | `encode.py:8` | ✅ Corrigé |
| S2 | 🔴 | Encodage **phrase par phrase** en boucle Python → inexploitable à l'échelle | `encode.py:27-37` | ✅ Corrigé |
| S3 | 🔴 | `last_hidden_state.mean(dim=1)` = mean-pool **sans masque** → erreur dès qu'on batche des séquences de longueurs différentes | `encode.py:36` | ✅ Corrigé |
| S4 | 🟠 | Aucun **décodeur** embedding→texte → impossible d'évaluer les générations LCM en texte | — | ✅ Implémenté |
| S5 | 🟡 | `split_long_text` rejetait tous les chunks ≤ 50 caractères → phrases courtes perdues | `encode.py:11-16` | ✅ Corrigé |

### 2.3 Pourquoi rester HF-natif et ne pas utiliser le package `sonar-space`

Le package officiel `sonar-space` de Meta dépend de `fairseq2`, une librairie C++ avec :
- **Pas d'ABI stable** : exige un appariement exact version PyTorch / version CUDA.
- Installations régulièrement cassées par les mises à jour PyTorch.
- Erreurs `load_nllb_tokenizer` fréquentes sur les environnements homelab.

Deux ports HF permettent d'utiliser SONAR sans `fairseq2` :
- **Encodeur** : `cointegrated/SONAR_200_text_encoder_hf` — compatible `from_pretrained` standard
- **Décodeur** : `raxtemur/SONAR_200_text_decoder` — port communautaire basé sur M2M100

Ces deux modèles sont chargés via `lcm_explo/m2m_100/` et `transformers.NllbTokenizer`, sans dépendance C++ supplémentaire.

### 2.4 Correctifs appliqués

**Import spaCy paresseux (S1)** : remplacement du `nlp = spacy.load(...)` global par un `@lru_cache def _get_nlp()` — appelé uniquement quand la segmentation spaCy est réellement nécessaire.

**Encodage batché + masked mean-pool (S2, S3)** : nouvelle fonction `encode_sentences_sonar(sentences, device, tokenizer, model, batch_size=32)` :

```python
inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
hidden = outputs.last_hidden_state          # (B, T, 1024)
mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
```

**Décodeur SONAR (S4)** : `SonarTextDecoder` (`lcm_explo/adapters/sonar_decoder/`) charge `raxtemur/SONAR_200_text_decoder` et décode via `M2M100ForConditionalGeneration.generate(encoder_outputs=...)`.

---

## 3. Base-LCM

### 3.1 Rappel du papier

Le papier §3.1 décrit une baseline de régression MSE (Base-LCM) :
1. Normaliser les embeddings par une **statistique per-dimension** figée sur le corpus.
2. Un Transformer décodeur causale prédit le prochain concept en espace **normalisé**.
3. La loss MSE (ici RMSE) est calculée en espace **normalisé**.

Le papier conclut (§5) que Base-LCM souffre de **mode-averaging** : la MSE pousse le modèle vers la moyenne des concepts suivants plausibles, produisant des sorties floues. C'est la variante la plus faible ; la diffusion est recommandée.

### 3.2 Constats

| # | Sév. | Constat | Fichier | Statut |
|---|---|---|---|---|
| L1 | 🔴 | `StandardScaler` normalisait avec un **scalaire** (`zeros(1)`/`ones(1)`, `x.mean()` global) — les 1024 dims SONAR ont des échelles très différentes | `base_lcm.py` | ✅ Corrigé |
| L2 | 🔴 | Normalisation **EMA en ligne** pendant l'entraînement au lieu de stats **fixes pré-calculées** — `pass_counter` croissait sans fin | `base_lcm.py` | ✅ Corrigé |
| L3 | 🟠 | Loss calculée en espace SONAR **brut** (après dénormalisation) → dimensions à grande échelle dominent le gradient | `train_base_lcm.py` | ✅ Corrigé |
| L4 | 🟠 | `RotaryPositionalEmbedding` implémentait en fait un **PE sinusoïdal absolu**, pas du RoPE — nom trompeur | `nn/rotary_*/` | ✅ Renommé |
| L5 | 🟡 | QK-norm appliqué sur `embed_dim` entier **avant** split en têtes — devrait être par tête | `nn/multihead_attention/` | ✅ Corrigé |
| L6 | 🟡 | `BaseLCMDecoder.layer_norm` codé en dur en `nn.LayerNorm` même si `norm_type="DyT"` | `base_lcm.py` | ✅ Corrigé |
| L7 | 🟡 | Dropout 0.3/0.2 — trop élevé pour une régression continue | `base_lcm.py` | ✅ Corrigé → 0.1 |
| L8 | 🟡 | `intermediate_size=4096` pour `hidden_size=2048` → ratio FFN 2× au lieu de 4× | Config | Documenté |

### 3.3 Correctifs appliqués

**Normalizer par-dimension figé (L1, L2)** : remplacement de `StandardScaler` par `Normalizer(nn.Module)` avec buffers `mean`/`std` de forme `(1024,)`. Les stats sont calculées une fois sur le corpus par `fit_normalizer_task` puis chargées via `load_normalizer_stats(path)`. Les buffers sont inclus dans le `state_dict` → checkpointés automatiquement.

```python
class Normalizer(nn.Module):
    def load_stats(self, mean, std): ...
    def normalize(self, x): return (x - self.mean) / (self.std + self.eps)
    def denormalize(self, x): return x * (self.std + self.eps) + self.mean
```

**Loss en espace normalisé (L3)** :
```python
y_pred = self(x, padding_mask)                   # normalized space
y_norm = self.model.normalizer.normalize(y)       # normalize target
loss = self.loss(y_pred, y_norm, padding_mask)
```

**PE renommé (L4)** : le répertoire `rotary_positional_mbeddings/` est renommé en `sinusoidal_positional_embeddings/`, la classe `RotaryPositionalEmbedding` → `SinusoidalPositionalEmbedding`. Le vrai RoPE (rotation des clés/requêtes) est hors scope.

**QK-norm par tête (L5)** : `q_proj` et `k_proj` deviennent de simples `nn.Linear`. Le norm s'applique sur `head_dim` **après** le reshape `(B, T, H)` → `(B, num_heads, T, head_dim)` :
```python
q = self.q_proj(query)
q = q.view(B, T, num_heads, head_dim).transpose(1, 2)
q = self.q_norm(q) * self.scaling   # LayerNorm(head_dim) per-head
```

**`_make_norm` helper (L6)** : helper centralisé qui respecte `norm_type` dans tous les layers du decoder, y compris la norm finale.

**Sérialisation simplifiée** : `_serialize_module` dans `checkpoints.py` ne renvoie plus que `model_bytes` (safetensors) — le `scaler.json` est supprimé, les stats normalizer voyagent dans le `state_dict`.

---

## 4. One-Tower LCM (diffusion)

### 4.1 Rappel du papier

Le papier §3.3 décrit One-Tower LCM comme un modèle de **diffusion dans l'espace SONAR normalisé** : le décodeur reçoit à la fois le contexte et le concept cible bruité, et prédit le concept débruité. L'implémentation de référence utilise un processus de diffusion **variance-preserving** avec sampler itératif.

Nous utilisons la variante plus simple **flow-matching** (interpolation linéaire + prédiction de vélocité), qui est numériquement plus stable et donne des résultats comparables.

### 4.2 Constats

| # | Sév. | Constat | Fichier | Statut |
|---|---|---|---|---|
| L9 | 🔴 | `_add_noise = x + noise_level·ε` — bruit additif **non variance-preserving** ; pas un processus de diffusion valide | `one_tower_lcm.py` | ✅ Corrigé |
| L10 | 🔴 | Aucun **masque causal** en self-attention ni cross-attention → attention bidirectionnelle → fuite d'information future | `one_tower_lcm.py` | ✅ Corrigé |
| L11 | 🟠 | Buffer `timesteps` jamais utilisé ; aucun **sampler itératif** pour l'inférence | `one_tower_lcm.py` | ✅ Corrigé |
| L12 | 🟡 | `OneTowerLCM` possédait sa propre `RMSELoss` (incohérent avec `BaseLCM`) | `one_tower_lcm.py` | ✅ Supprimé |

### 4.3 Correctifs appliqués

**Flow-matching (L9)** : processus `x_t = (1-t)*x₀ + t*ε`, vélocité cible `v = ε - x₀`, `t ∼ U(0,1)`. À l'inférence : intégration Euler de `t=1` (bruit pur) à `t=0` (concept propre).

```python
def _interpolate(self, x0, t):
    eps = torch.randn_like(x0)
    t_b = t.view(-1, 1, 1)
    x_t = (1.0 - t_b) * x0 + t_b * eps
    v_target = eps - x0
    return x_t, v_target
```

**Masques causaux (L10)** : helper `_causal_mask(seq_len, device)` produit un masque additif upper-triangulaire `(-inf)` appliqué à la self-attention **et** à la cross-attention de chaque layer décodeur.

**Sampler `sample()` (L11)** : intégration ODE de `t=1` à `t=0` par pas d'Euler :

```python
@torch.no_grad()
def sample(self, input_concepts, steps=None):
    ctx = self.normalizer.normalize(input_concepts)
    x = torch.randn_like(ctx)
    for i in range(len(ts) - 1, 0, -1):
        t = ts[i].expand(x.shape[0])
        dt = ts[i] - ts[i - 1]
        x = x - dt * self._denoise(x, ctx, t)
    return self.normalizer.denormalize(x)
```

**Loss cohérente (L12)** : `OneTowerLCM.forward` renvoie `{"loss": F.mse_loss(pred_v, v_target)}`. Le training module lit ce champ directement — pas de `RMSELoss` dans le modèle.

**Pourquoi diffusion > Base-LCM** : la MSE force le modèle vers la **moyenne** des concepts suivants plausibles (mode-averaging) — sortie sémantiquement floue. La diffusion modélise la distribution complète des suites possibles et peut échantillonner des prédictions cohérentes et variées.

---

## 5. Pipeline de données

### 5.1 Constats

| # | Sév. | Constat | Statut |
|---|---|---|---|
| D1 | 🔴 | `_data_tasks.py` non fonctionnel : `SONAR_MODEL_NAME` inexistant, `InMemorySplitter(splitter_type=...)` stub de test | ✅ Réécrit |
| D2 | 🟠 | Aucun adaptateur `SaT` implémentant l'interface `TextSplitter` | ✅ `SaTSplitter` créé |
| D3 | 🟠 | Trois scripts quasi identiques (setup SONAR/splitter dupliqué) | ✅ Factorisé |
| D4 | 🟡 | Dataset Wikipedia incohérent entre scripts (`20220301` vs `20231101`) | ✅ Unifié sur `20231101.en` |
| D5 | 🟡 | Chargement non-streaming → casse à 6M articles | ✅ `streaming=True` |
| D6 | 🟡 | IDs documents incohérents (`title` vs `id`) | ✅ Unifié sur `row["id"]` |
| D7 | 🟠 | Aucune étape de **fit du normaliseur** — artefact `mean/std` par-dim attendu par le LCM | ✅ `fit_normalizer_task` ajouté |

### 5.2 Correctifs appliqués

**`SaTSplitter`** (`lcm_explo/adapters/splitter/sat/`) : implémente `TextSplitter` via `wtpsplit.SaT` avec import différé. Rend le pipeline indépendant de spaCy.

**`encode_article_batch_task`** : charge `M2M100EncoderModel` + `NllbTokenizer` de HF, route l'encodage par `encode_sentences_sonar`. Aucune dépendance sur `InMemorySplitter` ou `SONAR_MODEL_NAME`.

**`fit_normalizer_task`** : calcule mean/std **par dimension** sur tous les `.pt` du corpus en deux passes (streaming pour éviter les OOM), persiste `normalizer.pt` :

```python
torch.save({"mean": mean.float(), "std": std.float(), "count": n}, out_path)
```

Ce fichier est chargé automatiquement par `train_base_lcm_flow` et `train_one_tower_lcm_flow` en début d'entraînement.

---

## 6. Plan de remédiation (résumé)

| Phase | Titre | Statut |
|---|---|---|
| 0 | Document d'audit (ce fichier) | ✅ |
| 1.1 | Import spaCy paresseux | ✅ |
| 1.2 | `SaTSplitter` | ✅ |
| 1.3 | Encodage batché + masked mean-pool | ✅ |
| 1.4 | Constantes HF SONAR | ✅ |
| 1.5 | Réécriture `_data_tasks.py` | ✅ |
| 1.6 | `fit_normalizer_task` | ✅ |
| 1.7 | `_data_flows.py` streaming + normaliser | ✅ |
| 1.8 | `SonarTextDecoder` | ✅ |
| 2.1-2.2 | `Normalizer` par-dim, PreNet/PostNet/BaseLCM | ✅ |
| 2.3 | Loss en espace normalisé | ✅ |
| 2.4 | `checkpoints.py` simplifié | ✅ |
| 2.5 | Renommage PE sinusoïdal | ✅ |
| 2.6 | `_make_norm` helper | ✅ |
| 2.7 | Dropout abaissé à 0.1 | ✅ |
| 2.8 | QK-norm par tête | ✅ |
| C | Presets de taille de modèle + singledispatch | ✅ |
| 3 | One-Tower flow-matching, masques causaux, `sample()` | ✅ |
| 4 | Tests + validation | En cours |

### Hors scope (travaux futurs)

- **RoPE réel** (rotation Q/K) : le PE sinusoïdal renommé suffit pour la baseline.
- **Two-Tower diffusion LCM** : extension après validation du One-Tower.
- **Évaluation textuelle SONAR** : le décodeur `SonarTextDecoder` est implémenté mais son API doit être vérifiée à l'exécution contre la model card de `raxtemur/SONAR_200_text_decoder`.
