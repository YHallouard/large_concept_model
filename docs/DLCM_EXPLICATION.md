# DLCM — Documentation explicative

> Guide de compréhension du **Dynamic Large Concept Model** tel qu'implémenté dans
> `lcm_explo`. Objectif : que tu comprennes **exactement** ce que fait chaque étage,
> **pourquoi**, et comment ça se calcule. Basé sur `docs/DLCM.pdf` (ByteDance,
> arXiv:2512.24617).

---

## 0. La confusion à lever d'abord : « on prédit le prochain concept, donc comment on décode ? »

C'est **la** question à clarifier, parce que DLCM ne fonctionne **pas** comme le Base-LCM.

### Base-LCM (ce que tu connais)

```
phrases → [SONAR encoder gelé] → embeddings de phrases
        → [LCM] prédit le PROCHAIN embedding de phrase (régression)
        → [SONAR decoder gelé] → texte
```

Ici **la sortie du modèle EST un concept** (un vecteur SONAR de la phrase suivante).
On a besoin d'un décodeur SONAR séparé pour retransformer ce vecteur en texte.
La « prédiction » vit dans l'espace des concepts.

### DLCM (ce qu'on vient d'implémenter)

```
tokens → [encodeur] → états par token H
       → [segmentation] → concepts C (compressés, M ≪ L)
       → [backbone] raisonnement profond sur C → Z
       → [décodeur] cross-attention(H, Z) → états par token
       → [LM head] → logits du PROCHAIN TOKEN
```

**DLCM prédit le prochain _token_, comme GPT.** Les concepts ne sont **jamais** la
sortie : ce sont un **espace de raisonnement interne**, un goulot d'étranglement
compressé. On **ne décode jamais un concept vers du texte**. Il n'y a **pas** de
décodeur SONAR, pas d'embedder externe.

Le rôle des concepts : faire tourner le **raisonnement coûteux** (le backbone, qui
contient la majorité des paramètres) sur une séquence **courte** de M concepts au lieu
de L tokens — puis « redescendre » vers les tokens via la cross-attention pour garder
la précision token par token.

> **Analogie.** Tu lis une phrase mot à mot (encodeur), tu la résumes en quelques idées
> (concepts), tu **réfléchis sur les idées** (backbone), puis tu écris le mot suivant en
> t'appuyant sur ces idées (décodeur + LM head). Ce que tu produis, c'est **le mot
> suivant**, pas l'idée.

### Où se passe la « prédiction » exactement ?

Décalage next-token standard : à la position `t`, le modèle produit la distribution du
token `t+1`.

```
input_ids :  [<s>  The  cat  sat  on   the  mat]     (positions 0..6)
labels    :  [The  cat  sat  on   the  mat  .  ]     (= input décalé de 1)
             pos0 prédit "The", pos1 prédit "cat", ...
```

C'est exactement l'entraînement d'un LM causal. Les concepts sont juste un **contexte
de raisonnement** injecté dans le décodeur.

---

## 1. Tokenizer et « embedder »

| Question | Réponse | Pourquoi |
|---|---|---|
| **Tokenizer** | **GPT-2** (`gpt2`, vocab 50 257) via `transformers` | Léger, standard, anglais. La table d'embeddings reste raisonnable pour un modèle small. EOS = 50256 sépare les documents. |
| **Embedder** | **Aucun embedder externe.** Le modèle apprend sa propre table `nn.Embedding(50257, d_token)` | Différence clé avec Base-LCM : pas de SONAR, pas de vecteurs pré-calculés. Les « concepts » sont des moyennes **apprises** d'états internes, pas des embeddings SONAR. |

Donc : **DLCM = tokens bruts → embeddings appris → … → tokens**. Tout est appris de bout
en bout, il n'y a rien de gelé. Base-LCM avait un encodeur ET un décodeur SONAR gelés ;
DLCM n'en a aucun.

Table d'embeddings **liée** (tied) au LM head : `lm_head.weight = embedding.weight`
(un seul jeu de poids partagé pour projeter les tokens en vecteurs **et** les vecteurs en
logits). Économise ~25 M paramètres au small.

### 1.1 Est-ce efficace ? Le compromis vocab / warm-start

Deux limites réelles du choix GPT-2 :

1. **Vocab 50k = bon compromis à cette échelle.** Avec un tokenizer plus récent (Llama 3,
   128k), l'embedding pèserait ~65 M sur un budget 150 M (44 %) — trop lourd. GPT-2 reste à
   ~17 %. Mais son BPE (2019) compresse **mal le code** (pas de gestion spéciale de
   l'indentation, contrairement à StarCoder/Llama 3) — pénalisant vu les 25 % de code dans
   le corpus (plus de tokens pour la même quantité d'information).
2. **Init aléatoire = démarrage à froid.** Sur un budget de tokens limité (POC 1–2 Md), le
   modèle passe du temps à réapprendre des statistiques déjà connues (fréquences, proximité
   sémantique des tokens) que GPT-2 encode déjà dans sa table d'embeddings.

**Solution implémentée : warm-start par PCA** (`domain/usecases/warm_start.py`). Au lieu
d'une init aléatoire, on réduit la table d'embeddings pré-entraînée de GPT-2 (`768 → d_token`)
par **PCA** (analyse en composantes principales) et on l'utilise comme point de départ.

**Ce n'est PAS un embedder externe à l'inférence** — c'est juste une meilleure *init* des
poids. Une fois copiée, la table `nn.Embedding` redevient un paramètre entraînable comme les
autres ; GPT-2 n'est plus jamais chargé après l'init.

**Calcul (PCA via SVD tronquée)** — matrice source `A ∈ ℝ^{50257×768}` (embeddings GPT-2) :

```
A_centrée = A − mean(A)                          (centrage, comme une init aléatoire ~N(0,·))
A_centrée ≈ U · diag(S) · Vᵀ                      (SVD tronquée, torch.pca_lowrank, q = d_token)
projection = A_centrée · V ≈ U · S                (raccourci : pas besoin de refaire le produit)
projection_finale = projection / std(projection) · initializer_range
```

- `U ∈ ℝ^{50257×d_token}`, `S ∈ ℝ^{d_token}`, `V ∈ ℝ^{768×d_token}` : on garde les
  `d_token` **composantes principales** — les directions de plus grande variance de l'espace
  d'embedding GPT-2 (celles qui séparent le mieux les tokens entre eux).
- **Rescaling final** : la magnitude naturelle de GPT-2 ne correspond pas forcément à celle
  attendue par le reste du réseau (`initializer_range = 0.02`) ; on renormalise pour garder le
  flux résiduel stable dès le premier pas de gradient.

**Garde-fou** : ça ne s'applique que si `vocab_size` du modèle correspond exactement à celui
de la source (les ids de tokens doivent se correspondre 1:1). Avec `gpt2` (50 257) et notre
tokenizer (`gpt2`), c'est automatique. Si les vocabs diffèrent, la fonction **saute
silencieusement** (avertissement loggé) plutôt que de planter l'entraînement.

**Activation** (opt-in, désactivé par défaut) :
```python
DLCMTrainingConfig(warm_start_embedding=True, warm_start_source="gpt2")
```

---

## 2. Vue d'ensemble

```
                        ┌───────────────────────────────────────────────┐
 input_ids (B, L)       │                    DLCM                        │
      │                 │                                               │
      ▼                 │                                               │
 ┌─────────┐   H (B,L,d_token)                                          │
 │ ENCODEUR│──────────────┬───────────────────────────────┐            │
 │ causal  │              │                               │            │
 └─────────┘              ▼                               │            │
                    ┌───────────┐  b (frontières)         │ (skip:     │
                    │SEGMENTATION│─────────┐              │  H sert    │
                    │ + pooling  │         │              │  aussi de  │
                    └───────────┘         │              │  requête   │
                          │ C (B,M,d_concept)             │  au        │
                          ▼                │              │  décodeur) │
                    ┌───────────┐          │              │            │
                    │ BACKBONE  │          │              │            │
                    │ causal    │          │              │            │
                    │ (gros)    │          │              │            │
                    └───────────┘          │              │            │
                          │ Z (B,M,d_concept)             │            │
                          ▼                ▼              ▼            │
                    ┌────────────────────────────────────────┐        │
                    │ DÉCODEUR : cross-attn(H, Z) causale     │        │
                    │  + couches causales + RMSNorm           │        │
                    └────────────────────────────────────────┘        │
                          │ (B, L, d_token)                            │
                          ▼                                            │
                    ┌───────────┐                                     │
                    │  LM HEAD  │ (lié à l'embedding)                  │
                    └───────────┘                                     │
                          │ logits (B, L, vocab)                      │
                          └───────────────────────────────────────────┘
```

Deux « largeurs » (widths) différentes :
- **`d_token`** (512 au small) : côté tokens, encodeur + décodeur. Léger, opère sur L positions.
- **`d_concept`** (1024 au small) : côté concepts, le backbone. **Plus large et plus profond**, mais opère sur M ≈ L/4 positions.

C'est le cœur de l'idée : **déplacer le calcul des tokens (redondants) vers les concepts
(denses)**.

---

## 3. Parcours détaillé, avec un exemple

Prenons `L = 7` tokens : `[<s>, The, cat, sat, on, the, mat]`.

### 3.1 Encodage

`H = Encoder(embedding(input_ids))` : transformer **causal** (chaque token ne voit que
le passé). Sortie `H ∈ ℝ^{7×d_token}` — un vecteur contextuel par token.

### 3.2 Détection de frontières (§3.3.1) — calcul

On projette chaque état dans un espace « scan » de dimension `d_scan` :

```
q_t = W_q · h_t        k_t = W_k · h_t        (W_q, W_k : d_token → d_scan)
```

Puis la **probabilité de frontière** mesure la dissimilarité locale :

```
        1 - cos(q_{t-1}, k_t)
p_t  =  ────────────────────           avec p_1 = 1 forcé
                 2
```

**Intuition du calcul** (`cos ∈ [-1, 1]`) :

| cos(q_{t-1}, k_t) | p_t | interprétation |
|---|---|---|
| ≈ +1 (états très proches) | ≈ 0 | on continue le même concept |
| ≈ 0 (orthogonaux) | 0.5 | ambigu |
| ≈ −1 (opposés) | ≈ 1 | **rupture sémantique → nouvelle frontière** |

Exemple numérique : si `q_{t-1} = [1, 0]` et `k_t = [0, 1]`, alors `cos = 0` → `p_t = 0.5`.
Si `k_t = [-1, 0]`, `cos = -1` → `p_t = 1`.

**Passage au discret** (frontière `b_t ∈ {0,1}`) :
- **Entraînement** : on « aiguise » `p` par une température `α = 0.5`, puis on tire au sort
  `b_t ∼ Bernoulli(p_sharp)`. Aiguisage :
  ```
  p_sharp = p^{1/α} / ( p^{1/α} + (1-p)^{1/α} )
  ```
  (α < 1 pousse vers 0 ou 1). **`b` est échantillonné SANS gradient** — point crucial, voir §5.
- **Inférence** : seuil dur `b_t = [p_t ≥ 0.5]`.

Dans notre exemple, disons que la segmentation donne :
```
b = [1,  1,   0,   1,   0,  1,   0]
     <s> The  cat  sat  on  the  mat
```

### 3.3 Formation des concepts par pooling (§3.3.2) — calcul

`seg_id = cumsum(b) - 1` donne l'indice de segment (0-based) de chaque token :

```
b       = [1, 1, 0, 1, 0, 1, 0]
cumsum  = [1, 2, 2, 3, 3, 4, 4]
seg_id  = [0, 1, 1, 2, 2, 3, 3]     → M = 4 segments
```

Segments : `{<s>}`, `{The, cat}`, `{sat, on}`, `{the, mat}`.

Chaque concept = **moyenne** des états du segment, puis projection `W_up` (d_token → d_concept) :

```
c_k^raw = (1/|S_k|) · Σ_{t ∈ S_k} h_t          (moyenne)
c_k     = W_up · c_k^raw                        (projection vers d_concept)
```

Exemple : `C2^raw = (h_The + h_cat) / 2`.

Implémentation : `scatter_add_` accumule les `h_t` par `seg_id` puis divise par le compte.
**Cette opération est différentiable par rapport à `h`** → l'erreur de prédiction (CE)
remonte dans l'encodeur via le pooling. Mais `seg_id` (des entiers) n'a **aucun gradient**
→ la CE ne touche jamais `W_q`/`W_k`. (Voir §5.)

Résultat : `C ∈ ℝ^{4×d_concept}` — on est passé de 7 tokens à 4 concepts.

### 3.4 Backbone — raisonnement sur les concepts (§3.4)

`Z = Backbone(C)` : transformer **causal** profond, largeur `d_concept`, la majorité des
paramètres du modèle. Comme il ne voit que 4 positions au lieu de 7, son attention coûte
`O(M²)` au lieu de `O(L²)` → **c'est là qu'est l'économie de calcul**.

`Z ∈ ℝ^{4×d_concept}` : concepts « raisonnés ».

### 3.5 Décodage — LE point clé (§3.5)

Comment redescendre de 4 concepts raisonnés vers 7 prédictions de tokens ? Par une
**cross-attention causale** : chaque token `t` interroge (query) les concepts (key/value).

**Contrainte de causalité** : le token `t` ne peut voir que les concepts des segments
**terminés AVANT son propre segment**. Sinon fuite : le concept de son propre segment a
« moyenné » des tokens ≥ t (dont le label t+1) → la CE tricherait.

#### Comment on l'implémente : réplication décalée + BOS

On construit une banque de key/value où **l'indice k contient le concept c_{k-1}**
(l'indice 0 = un concept BOS appris) :

```
kv_bank = [ BOS,  C1,  C2,  C3 ]        (C4 volontairement absent, voir plus bas)
             0     1    2    3
```

Puis on **réplique** par token via `seg_id` :

```
seg_id  = [0,   1,   1,   2,   2,   3,   3]
z_rep   = kv_bank[seg_id]
        = [BOS, C1,  C1,  C2,  C2,  C3,  C3]
           <s>  The  cat  sat  on   the  mat
```

Enfin **attention causale standard** (token t voit les positions ≤ t) sur `z_rep` :

```
                 clés répliquées (z_rep) →
              BOS  C1   C1   C2   C2   C3   C3
 queries    ┌───────────────────────────────────
   <s>  t0  │ ■    ·    ·    ·    ·    ·    ·      → voit {BOS}
   The  t1  │ ■    ■    ·    ·    ·    ·    ·      → voit {BOS, C1}
   cat  t2  │ ■    ■    ■    ·    ·    ·    ·      → voit {BOS, C1}
   sat  t3  │ ■    ■    ■    ■    ·    ·    ·      → voit {BOS, C1, C2}
   on   t4  │ ■    ■    ■    ■    ■    ·    ·      → voit {BOS, C1, C2}
   the  t5  │ ■    ■    ■    ■    ■    ■    ·      → voit {BOS, C1, C2, C3}
   mat  t6  │ ■    ■    ■    ■    ■    ■    ■      → voit {BOS, C1, C2, C3}
            └───────────────────────────────────
              ■ = attendu     · = masqué (futur)
```

**Vérifions que c'est correct.** Prenons le token `cat` (t2, dans le segment 1). Il
prédit `sat`. Sous le masque causal il atteint `z_rep[0..2] = {BOS, C1, C1}`. Or `C1` =
concept du segment 0 (`{<s>}`), **terminé avant** son segment. Il ne voit **PAS** `C2`
(= pool de `The`+`cat`, qui contient `cat` lui-même **et** dépendrait de tokens futurs du
segment). ✅ Pas de fuite.

**Pourquoi C4 est absent de la banque ?** `C4` = pool de `{the, mat}`, le dernier segment.
Il ne servirait qu'à prédire un token **après** `mat` — hors séquence. Donc jamais utilisé.
C'est correct et voulu.

**Pourquoi le BOS ?** Le tout premier token (`<s>`) n'a aucun concept terminé avant lui.
Sans BOS, sa ligne d'attention serait entièrement masquée → `NaN` dans SDPA. Le BOS est un
concept appris qui garantit qu'aucune ligne n'est vide.

Formule complète (Éq. 14 du papier), avec résiduel :

```
Ψ(H, Z) = softmax( Q Kᵀ / √d_head + masque ) V · W_O  +  H
          où Q = H·W_Q (depuis les tokens),  K,V = z_rep·W_{K,V} (depuis les concepts)
```

Q vient de l'espace **token** (`d_token`), K/V de l'espace **concept** (`d_concept`), tous
deux projetés vers une dimension de tête commune `d_head`. Le `+ H` réinjecte l'info
token brute (résiduel).

Après la cross-attention : quelques couches de transformer causal (largeur `d_token`),
RMSNorm final, puis LM head → logits.

### 3.6 Concept smoothing (§3.5.1)

Petit détail avant la cross-attention : on « lisse » les concepts raisonnés `Z` avec une
**conv causale** sur l'axe des concepts (`Z̃ = Smoothing(Z)`) pour atténuer les artefacts
aux frontières de segment. **Causale obligatoirement** : un noyau symétrique mélangerait
`z_{k+1}` dans `z̃_k` et rouvrirait la fuite qu'on vient de fermer.

---

## 4. Récapitulatif du flux (schéma compact)

```
 tokens ──emb──► H ──┬──────────────────────────────────┐
                     │                                  │ (query)
              seg_id │                                  │
                     ▼                                  ▼
   H ──pool(seg)──► C ──backbone──► Z ──smooth──► Z̃ ──cross-attn──► H' ──►LM head──► logits
                     │                              (key/value)
      W_q,W_k ──► p ─┴─► aux loss (compression)
```

- Chemin **CE** (prédiction) : `H → C → Z → Z̃ → cross-attn → LM head`.
- Chemin **aux** (compression) : `H → p → aux loss`. **Séparé**, ne passe pas par la CE.

---

## 5. Les blocs, en détail

### 5.1 Blocs **réutilisés** (déjà présents, non recréés)

| Bloc | Fichier | Ce qu'il apporte à DLCM | Pourquoi réutilisé tel quel |
|---|---|---|---|
| `SinusoidalPositionalEmbedding` | `nn/sinusoidal_positional_embeddings` | Encodage positionnel additif (Vaswani) pour l'encodeur (positions de tokens) **et** le backbone (positions de concepts) | Générique, dépend juste de `d_model` |

### 5.1bis Blocs **volontairement PAS réutilisés** (et pourquoi)

| Bloc existant | Pourquoi on ne l'a **pas** pris | Ce qu'on a fait à la place |
|---|---|---|
| `QKNormedMultiheadAttention` (`nn/multihead_attention`) | (1) fait une **LayerNorm** par tête ; le papier veut une **RMSNorm**. (2) matérialise la matrice `(B,H,L,L)` en softmax manuel → **OOM** à L=1024 sur 11 Go. (3) ne gère **pas** `q_dim ≠ kv_dim`, indispensable pour la cross-attention token↔concept. | Nouveau `QKRMSNormSDPAAttention` basé sur `F.scaled_dot_product_attention` |
| `DyT` (`nn/dynamic_tanh`) | Normalisation alternative utilisée par Base-LCM ; le papier DLCM spécifie explicitement **RMSNorm** partout (QK-norm et pré-norm) | `nn.RMSNorm` de PyTorch |

### 5.2 Blocs **créés**

| Bloc | Rôle | Détail / calcul clé |
|---|---|---|
| `QKRMSNormSDPAAttention` | Attention réutilisable (self **et** cross) | RMSNorm par tête sur Q et K, puis SDPA. `q_proj: q_dim→H·head_dim`, `k/v_proj: kv_dim→H·head_dim`, `out_proj→q_dim`. La sortie vit dans l'espace **query**. |
| `BoundaryDetector` | Calcule `p_t` et échantillonne `b_t` | Modes `learned` (avec `W_q,W_k`) ou `rule` (cosinus direct sur `h`, plan B stable). `sample()`/`threshold()` sous `no_grad` → **`b` sans gradient**. |
| `SegmentMeanPooling` | Moyenne les tokens par segment | `scatter_add_` différentiable/`h`, `seg_id` non différentiable. Renvoie aussi `concept_mask` (padding des M variables). |
| `CausalConceptCrossAttention` | Le décodage token↔concept | Réplication décalée + BOS (§3.5), masque causal L×L. **Le bloc le plus délicat** (non-fuite). |
| `CausalConceptSmoothing` | Lisse les concepts raisonnés | Conv depthwise **causale** (padding gauche) + Linear + résiduel + RMSNorm. |
| `BoundaryRatioLoss` | Loss de compression (« Global Parser ») | Voir §6. F suivi en **EMA** pour rester global sous accumulation de gradient. |

### 5.3 Le découplage des gradients (le point subtil)

```
             ┌──────────── CE (cross-entropy next-token) ───────────┐
             │                                                       │
   embedding ← encodeur ← [pool: scatter_add] ← C ← backbone ← Z ← décodeur ← LM head
             ▲                                                       
             │ gradient OK (via la moyenne des h_t)                  
                                                                     
   W_q, W_k  ←──── SEUL chemin : G = mean(p) dans la loss aux ───────┘
             ▲
             │  b échantillonné SANS gradient ; seg_id entier sans gradient
```

**Pourquoi ?** (§3.3, §8.1 du papier). Si la CE pouvait entraîner `W_q`/`W_k`, elle
apprendrait à **ne plus segmenter** (moins de compression = moins de perte d'info = CE plus
faible). La CE écraserait la loss aux. En coupant ce chemin, la segmentation est pilotée
**uniquement** par la loss de compression. Vérifié par un test : après `ce.backward()` seul,
`W_q.grad` est nul ; il n'apparaît qu'avec la loss aux.

---

## 6. Les pertes — calculs

```
L = L_CE  +  λ · L_aux            (λ = 0.03 par défaut)
```

### 6.1 `L_CE` — cross-entropy next-token

Standard : `cross_entropy(logits[:, :, :].view(-1, V), labels.view(-1))`. C'est **la**
tâche : prédire le token suivant. Rien d'exotique.

### 6.2 `L_aux` — compression vers un ratio cible R (Éq. 10)

```
         R                                                  1
L_aux = ─── · [ (R-1)·F·G + (1-F)·(1-G) ]  −  1     minimum à F = G = ─
        R-1                                                          R
```
- `G = mean(p)` : taux de frontière **attendu** (continu) → **seul chemin de gradient** vers `W_q,W_k`.
- `F = mean(b)` : taux de frontière **réel** (discret, détaché).
- `R = 4` : on veut 1 frontière tous les 4 tokens en moyenne (ratio de compression 4×).

**Calcul du minimum** (R = 4) :

| F = G | `L_aux` | commentaire |
|---|---|---|
| 0.25 (= 1/R) | `4/3·[3·(1/16) + (3/4)²] − 1 = 4/3·(12/16) − 1 = 1 − 1 =` **0** | ✅ cible atteinte |
| 1.0 (tout est frontière) | `4/3·[3 + 0] − 1 =` **3** | pénalisé fort (aucune compression) |
| 0.0 (aucune frontière) | `4/3·[0 + 1] − 1 =` **1/3** | pénalisé (tout en 1 concept) |

La loss est **minimale exactement à 1/R** et croît de chaque côté → elle pousse la
compression moyenne vers 4×, tout en laissant fluctuer localement (code dense ↔ prose).

### 6.3 Le « Global Parser » et l'astuce EMA

Le papier calcule F et G sur **tout le batch global** (§8.2) — pas par séquence — pour
laisser le modèle compresser plus le code répétitif et moins la prose dense. Sous
**accumulation de gradient** (micro-batch 4 × 8), un micro-batch ne voit qu'1/8 des tokens.

Comme `L_aux` est **linéaire en F** et que F y entre **détaché**, on maintient F en **EMA**
inter-micro-batches (momentum 0.99) et on garde G local. Résultat : le gradient est
**identique** à celui d'un F global, sans réécrire la boucle d'optimisation. (Buffer
`f_ema` persistant → survit au resume.)

---

## 7. Combien de paramètres ? (small)

Config small : `d_token=512, d_concept=1024, enc=4, backbone=8, dec=2, ffn×4, vocab=50257`.

| Composant | Calcul | Params |
|---|---|---|
| Embedding (lié au LM head) | `50257 × 512` | 25.7 M |
| Encodeur (4 couches d_token) | par couche : attn `4·512² = 1.05M` + FFN `2·512·2048 = 2.1M` ≈ 3.15M ; ×4 | 12.6 M |
| Segmenteur | `W_q,W_k : 2·512·128` + `W_up : 512·1024` | 0.65 M |
| **Backbone (8 couches d_concept)** | par couche : attn `4·1024² = 4.2M` + FFN `2·1024·4096 = 8.4M` ≈ 12.6M ; ×8 | **100.6 M** |
| Décodeur | smoothing `~1M` + cross-attn `~1.6M` + 2 couches `6.3M` | ~8.9 M |
| LM head | lié → 0 supplémentaire | 0 |
| **Total** | | **≈ 149 M** |

**Observation clé** : le backbone (100 M) domine — c'est voulu. On concentre la capacité
sur le raisonnement conceptuel, qui tourne sur M ≈ L/4 positions. (Mesuré : 148.7 M.)

Au tiny (`d_token=256, d_concept=512`) : ≈ 30 M total, dont ~13 M pour l'embedding
(vocab 50k domine à petite échelle).

---

## 8. Combien de VRAM ? (small, L=1024, précision mixte)

| Poste | Estimation | Note |
|---|---|---|
| Poids + états AdamW | `149M × ~18 o ≈` **2.7 Go** | master fp32 + moments |
| Activations attention | ~150 Mo/échantillon | SDPA ne matérialise pas la matrice L² |
| **Logits + CE** | `4 × 1024 × 50257 × 4 o ≈` **0.8 Go** (batch 4) | poste **dominant**, upcast fp32 pour la CE |
| **Total (micro-batch 4)** | **≈ 5–6 Go** (pire cas M = L) | tient dans 11 Go |

Réglage recommandé : **micro-batch 4 × accumulate 8** (≈ 32k tokens/step effectif).
Échappatoires si besoin : CE par chunks, activation checkpointing du backbone.
Sur GPU Turing (2080 Ti, pas de bf16 rapide) : `precision="16-mixed"`.

---

## 9. Les phases d'entraînement

DLCM étant **token-level** (comme GPT), il se branche sur les recettes LLM standard —
contrairement à Base-LCM qui exigeait des astuces dans l'espace SONAR.

### Phase 1 — Pré-entraînement (implémenté)

- **Données** : mix web + code en streaming (`fineweb-edu` 75 % + `github-code-clean` 25 %),
  tokenisé GPT-2, packé en fenêtres de `seq_len` tokens (shards memmap uint16).
- **Objectif** : `L = L_CE + λ·L_aux`. Le modèle apprend **simultanément** à prédire le
  token suivant **et** à segmenter (via la loss aux).
- **Pourquoi ce mix ?** Des densités d'information très différentes (code structuré ↔ prose)
  forcent le détecteur de frontières à devenir **adaptatif** (§5 du papier).
- **Lancement** :
  ```
  prepare_dlcm_tokens_flow(PrepareDlcmTokensConfig(output_dir=..., total_tokens=1_000_000_000))
  train_dlcm_flow(TrainDLCMConfig(
      data=TokenDataConfig(tokens_dir=...),
      model=PresetDLCMModelSpec(size="small"),
      training=DLCMTrainingConfig(max_steps=30_000, micro_batch_size=4, accumulate_grad_batches=8),
  ))
  ```
- **À surveiller** (MLflow) : `train_ce` ↓ ; `train_compression_ratio` → ≈ 4 ; `train_aux`
  → ≈ 0 ; **alarmes de collapse** : F → 1 (pas de compression) ou F → 1/L (tout en 1
  concept). Profil `val_loss_pos_*` en **U** (Fig. 7) = le mécanisme marche.

### Phase 2 — SFT (fine-tuning instructionnel) — à venir

- **Données** : paires (instruction, réponse) — ex. LogiCoT, ROC Stories (déjà câblés pour
  Base-LCM), ou tout jeu instruct.
- **Objectif** : CE next-token **sur les tokens de réponse uniquement** (on masque le prompt
  avec `ignore_index`). On **garde la loss aux** (λ éventuellement réduit) pour ne pas casser
  la compression apprise ; possibilité de **geler le détecteur de frontières** ou de baisser
  son LR pour figer la segmentation.
- Rien de spécifique à DLCM ici : c'est du SFT LLM standard, la segmentation est un module
  interne transparent.

### Phase 3 — Post-training préférences (DPO/RLHF) — optionnel

- **DPO** sur des paires (préférée, rejetée) : perte de préférence standard sur les logits
  token-level. DLCM produisant des logits de tokens, **DPO s'applique directement** (pas
  besoin de récompense en espace SONAR comme le suggérait `TRAINING_STRATEGY.md` pour
  Base-LCM).
- Garder un petit `λ·L_aux` en régularisation pour préserver le ratio de compression.

### Résumé des différences de phase vs Base-LCM

| | Base-LCM | DLCM |
|---|---|---|
| Sortie | prochain **concept** (vecteur SONAR) | prochain **token** |
| Perte pretrain | MSE/RMSE en espace SONAR | **CE** next-token + aux compression |
| SFT | en espace concept | **CE token standard** (masque prompt) |
| Préférences | récompense cosinus SONAR | **DPO token standard** |
| Décodeur externe | SONAR gelé requis | **aucun** |

---

## 10. Où est le code

| Élément | Chemin |
|---|---|
| Modèle | `lcm_explo/domain/models/dlcm.py` |
| Blocs nn | `lcm_explo/domain/models/nn/{qk_rmsnorm_attention,boundary_detector,segment_pooling,concept_smoothing,causal_concept_cross_attention}/` |
| Loss aux | `lcm_explo/domain/models/_losses.py` (`BoundaryRatioLoss`) |
| Presets | `lcm_explo/domain/models/_presets.py` (`dlcm_config_for_size`) |
| Dataset | `lcm_explo/adapters/dataset/{packed_tokens,in_memory_tokens}/` |
| Entraînement | `lcm_explo/domain/usecases/train/train_dlcm.py` |
| Génération | `lcm_explo/domain/usecases/generate.py` |
| Warm-start embedding | `lcm_explo/domain/usecases/warm_start.py` |
| Workflows | `lcm_explo/workflows/{_inputs.py,_flows.py,_data_tasks.py,_data_flows.py}` |

---

## Annexe — glossaire des dimensions

| Symbole | Signification | Small |
|---|---|---|
| `B` | taille du (micro-)batch | 4 |
| `L` | longueur de séquence (tokens) | 1024 |
| `M` | nombre de concepts (variable, ≈ L/R) | ~256 |
| `d_token` | largeur côté tokens | 512 |
| `d_concept` | largeur côté concepts (backbone) | 1024 |
| `d_scan` | dimension de projection des frontières | 128 |
| `R` | ratio de compression cible | 4 |
| `V` | taille du vocabulaire (GPT-2) | 50257 |
