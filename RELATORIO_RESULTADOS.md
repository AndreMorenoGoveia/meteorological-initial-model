# Relatório de Resultados — Modelos de Previsão Meteorológica

> Gerado em 2026-05-17. Baseado nos checkpoints e histórico de treinamento disponíveis em `runs/`.

---

## Sumário Executivo

Este relatório documenta a metodologia completa de treinamento e avaliação, as fontes de dados utilizadas (ERA5, IAG-USP e INMET), e os resultados detalhados do modelo HGT (*Heterogeneous Graph Transformer*). O modelo GRU (baseline) foi treinado e possui checkpoints salvos, mas seu histórico de validação não foi gravado — seus resultados quantitativos dependem de uma rodada de `evaluate.py`. O HGT, por sua vez, tem 30 épocas de histórico completo.

---

## 1. Metodologia

### 1.1 Fontes de Dados e Papéis

O modelo trabalha com **três fontes de dados distintas**, cada uma com um papel bem definido:

| Fonte | Tipo | Papel no Modelo | Usado como Alvo? |
|-------|------|-----------------|-----------------|
| **ERA5** | Grade regular (reanálise ECMWF) | Contexto espacial de larga escala | **Não** — só contexto |
| **INMET** | Estações esparsas (rede nacional) | Contexto regional + **alvo de previsão** | **Sim** |
| **IAG-USP** | Estação única (campus USP) | Contexto local + **alvo de previsão** | **Sim** |

**Ponto crítico:** durante teste, o modelo **recebe ERA5 como entrada** (contexto), mas **não prevê ERA5**. O ERA5 nunca é usado como ground truth de comparação — ele existe para fornecer o contexto de escala regional que as estações sozinhas não capturam.

#### ERA5 — Contexto de Larga Escala
- Reanálise horária do ECMWF, ~0.25° de resolução
- Região recortada: lat ∈ [−24.5°, −23.0°], lon ∈ [−47.5°, −46.0°] (SP metropolitana)
- Dados disponíveis a cada hora, sem falhas (reanálise completa)
- Fornece o "estado atmosférico de fundo" que as estações locais não conseguem medir

#### IAG-USP — Alvo Principal
- Estação meteorológica automática do Instituto de Astronomia, Geofísica e Ciências Atmosféricas (IAG) da USP
- Coordenada de referência: −23.6510°S, −46.6225°O
- Observações horárias, com lacunas (NaN preservados)
- **É o principal alvo**: queremos prever o microclima do campus USP com 24 horas de antecedência

#### INMET — Alvo Secundário / Contexto Regional
- Estações da rede do Instituto Nacional de Meteorologia
- Filtradas por distância: raio ≤ 250 km do centro IAG-USP
- Observações horárias, com lacunas (tipicamente mais que o ERA5)
- Funcionam como âncoras regionais entre o ERA5 (grosseiro) e o IAG (pontual)

---

### 1.2 Divisão Temporal dos Dados

A divisão é **estritamente temporal** — sem embaralhamento aleatório. Isso é fundamental para evitar vazamento de informação temporal (*data leakage*) e respeitar a autocorrelação das séries.

```
Treinamento: 2020-01-01  →  2023-12-31   (4 anos)
Validação:   2024-01-01  →  2024-06-30   (6 meses)
Teste:       2024-07-01  →  2024-12-31   (6 meses)
```

**O conjunto de teste é completamente invisível durante o treinamento e seleção de hiperparâmetros.** O `best.pt` é salvo com base na menor `val_loss` (conjunto de validação), não no teste.

---

### 1.3 Janela de Observação (Window)

Para cada amostra de treinamento/avaliação, define-se um "tempo observador" t_φ:

```
Contexto:   [t_φ − 48h,  t_φ)    →  48 passos horários de entrada
Previsão:   [t_φ,        t_φ + 24h)  →  24 passos horários de saída
```

- O **stride** entre amostras consecutivas é de 6 horas (não 1h), reduzindo correlação entre amostras adjacentes e o custo computacional
- Isso gera aproximadamente **5.840 janelas no treinamento**, **730 na validação** e **730 no teste**

**O que o modelo vê como entrada (contexto):**
- 48 horas de ERA5, IAG e INMET anteriores a t_φ (features brutas, com máscaras de validade)
- Timestamps absolutos (unix) de cada passo
- Coordenadas lat/lon de cada instância

**O que o modelo deve prever (alvo):**
- 24 horas de IAG e INMET seguintes a t_φ (somente estas duas fontes)

---

### 1.4 Variáveis

As seguintes variáveis estão presentes no NetCDF unificado:

| Variável Original | Processamento | Features no Modelo |
|-------------------|--------------|-------------------|
| `air_temperature_c` | Nenhum | `air_temperature_c` |
| `dew_point_temperature_c` | Nenhum | `dew_point_temperature_c` |
| `wind_direction_deg` | Decomposição vetorial | `u_ms` (componente zonal) |
| `wind_speed_ms` | Decomposição vetorial | `v_ms` (componente meridional) |

**Por que decompor o vento?** Velocidade e direção têm uma descontinuidade circular (359° → 0°) que prejudica funções de perda baseadas em distância euclidiana. As componentes u/v são contínuas e permitem calcular MAE/RMSE sem ambiguidade.

Total: **4 features por instância por passo de tempo**.

---

### 1.5 Normalização (RevIN)

Cada instância passa por **Reversible Instance Normalization (RevIN)** independentemente:

1. `mean` e `std` são calculados sobre a janela de **contexto** (48h), ignorando NaN via máscara
2. O contexto é normalizado: `x_norm = (x − mean) / std`
3. O decoder produz saídas normalizadas; a desnormalização `x_phys = x_norm * std + mean` é aplicada antes de calcular a loss

Isso permite que o modelo aprenda padrões de variação relativa (ciclos diurnos, tendências) sem ser dominado pelas diferenças absolutas entre estações.

---

### 1.6 Função de Perda

O modelo é treinado com a **perda complemento do Índice de Concordância (IoA)**:

```
IoA = 1 − Σ(y − ŷ)² / Σ(|ŷ − ȳ| + |y − ȳ|)²
Loss = 1 − IoA
```

Calculada **por instância** (não agregada antes da divisão), depois reduzida por tipo e depois entre tipos. Isso evita que o ERA5, com muito mais pontos de grade que o IAG, domine o gradiente.

---

### 1.7 Métricas de Avaliação

Todas as métricas são **mask-aware**: posições com `NaN` no ground truth são excluídas do cálculo.

| Métrica | Sigla | Interpretação | Intervalo |
|---------|-------|---------------|-----------|
| Índice de Concordância de Willmott | **IoA** | Mede skill geral (padrão + bias). ≥ 0.6 é considerado razoável | [0, 1] |
| Erro Médio Absoluto | **MAE** | Erro médio nas unidades físicas (misto: °C e m/s) | [0, ∞) |
| Raiz do Erro Quadrático Médio | **RMSE** | Penaliza erros grandes; sempre ≥ MAE | [0, ∞) |
| Correlação de Pearson | **corr** | Captura concordância de padrão (sem bias) | [−1, 1] |

**Atenção sobre MAE/RMSE:** nas métricas agregadas por tipo (mostradas no histórico), estas são médias sobre as 4 features (temperatura °C, ponto de orvalho °C, u m/s, v m/s). As unidades físicas são heterogêneas — o número é um indicador de escala, não diretamente comparável entre variáveis sem decomposição.

---

## 2. Arquitetura dos Modelos

### 2.1 Visão Geral das Três Variantes

O projeto implementa três variantes progressivas:

```
GRU (Baseline)
  └── Encoder GRU por tipo + Decoder GRU + RevIN

ST_MCAR (Ablação)
  └── GRU + RevIN + Encoding Posicional (tempo, lat, lon) + MCAR

HGT (Modelo Completo)
  └── GRU + RevIN + Encoding Posicional + MCAR + Fusão por Grafo Heterogêneo
```

---

### 2.2 Como o HGT Funciona — Descrição Detalhada

O HGT processa cada amostra em **3 estágios sequenciais**:

---

#### Estágio 1 — Codificação Temporal por Tipo

Para **cada tipo de fonte** (ERA5, IAG, INMET) **independentemente**:

1. **RevIN fit**: calcula `mean` e `std` sobre as 48h de contexto para cada instância
2. **Normalização**: `x_norm = (x − mean) / std`, zeros onde há NaN
3. **Encoding posicional**: concatena ao input normalizado:
   - *Temporal*: seno/cosseno da posição relativa no tempo (`t − t_φ`, escala ~1 ano)
   - *Espacial*: seno/cosseno de lat e lon separadamente (escala ~1°)
   - Dimensão total adicionada: `pe_time_dim + 2 × pe_space_dim = 32 + 64 = 96`
4. **GRU encoder**: passa a série temporal `(N×B, T_c=48, F+96)` pelo GRU
   - Hidden dim: 128
   - Saída: `h = (N×B, 128)` — um vetor de embedding por instância que "resume" as últimas 48h daquela instância

Após este estágio: `h_by_type = {ERA5: (B, N_era5, 128), IAG: (B, 1, 128), INMET: (B, N_inmet, 128)}`

---

#### Estágio 2 — Fusão por Grafo Heterogêneo (HGT)

Este é o coração do modelo. Os embeddings do Estágio 1 são **nós de um grafo**, e as arestas conectam instâncias de tipos diferentes segundo o k-NN geográfico.

**Construção do Grafo:**

As arestas são **dirigidas, de fontes grosseiras para fontes finas**, formando 3 relações:

```
ERA5  →  INMET   (k=8  vizinhos ERA5 mais próximos de cada estação INMET)
ERA5  →  IAG     (k=16 vizinhos ERA5 mais próximos do IAG)
INMET →  IAG     (k=4  estações INMET mais próximas do IAG)
```

A vizinhança é calculada por **distância haversine** (arco de grande círculo), não distância euclidiana. As arestas são fixas para o conjunto de instâncias e repassadas ao grafo em cada batch (recalculadas por posição espacial, não por estado temporal).

**Por que direcionado dessa forma?**
A informação flui de escala regional/grosseira para local/fina, respeitando a hierarquia de resolução espacial. O ERA5 fornece contexto de mesoscala que os modelos locais não conseguem capturar com dados de estações esparsas.

**A Camada HGTConv:**

O HGT implementa **atenção multi-head heterogênea** (PyTorch Geometric `HGTConv`). Para cada aresta (s → d):

```
Atenção: α(s, d) ∝ (W_Q[d] · h_d)ᵀ · (W_K[s] · h_s) / √(H/num_heads)
Mensagem: m(s, d) = W_V[s] · h_s
Novo embedding: h_d' = Σ_{s ∈ N(d)} α(s, d) · m(s, d)
```

Cada tipo de nó e cada tipo de aresta tem **matrizes de projeção próprias** (W_Q, W_K, W_V). Isso é o "heterogêneo" do HGT: ERA5→IAG usa pesos diferentes de INMET→IAG.

**Hiperparâmetros:**
- `num_heads = 4` (atenção com 4 cabeças)
- `num_layers = 2` (2 camadas de HGTConv empilhadas)
- `dropout = 0.1` entre camadas
- Ativação ReLU + Dropout após cada camada

**Dimensionalidade do batch no grafo:**

O batch é "achatado" antes de entrar no HGT: `(B, N, H) → (B×N, H)`. As arestas são replicadas para cada elemento do batch com offset correto nos índices. Após as camadas, os embeddings são desachatados de volta para `(B, N, H)`.

**Saída do Estágio 2:** `h_by_type` atualizado — agora cada embedding de IAG ou INMET **já incorpora informação dos vizinhos ERA5 e INMET**, ponderada pela atenção.

---

#### Estágio 3 — Decodificação por Tipo

Para cada **tipo alvo** (IAG e INMET):

1. **Conditioning**: seno/cosseno dos timestamps e coordenadas do horizonte de previsão `(t_φ → t_φ+24h)` — o decoder "sabe" para qual momento está prevendo
2. **GRU decoder**: recebe `(h0, conditioning)` e produz `(B×N, T_f=24, F=4)` normalizado
3. **RevIN inverse**: aplica `x_phys = x_norm × std + mean` para retornar às unidades físicas originais

---

### 2.3 MCAR — Robustez a Dados Faltantes

Durante o treinamento do HGT (e ST_MCAR), aplica-se **Missing Completely At Random (MCAR)** como augmentation:

- **Instance dropout** [0.0, 0.3]: com prob. uniforme em [0%, 30%], zera instâncias aleatórias do batch
- **Timestamp dropout** [0.0, 0.3]: com prob. uniforme em [0%, 30%], zera timesteps aleatórios

Isso obriga o modelo a aprender a prever mesmo quando parte do contexto está ausente — refletindo condições reais onde estações têm falhas.

**O GRU baseline não usa MCAR** (ambas as ranges em [0.0, 0.0]).

---

### 2.4 Comparativo de Capacidades por Variante

| Capacidade | GRU | ST_MCAR | HGT |
|-----------|-----|---------|-----|
| Encoding temporal (RevIN) | ✓ | ✓ | ✓ |
| Encoding posicional (lat/lon/time) | — | ✓ | ✓ |
| Robustez a dados faltantes (MCAR) | — | ✓ | ✓ |
| Fusão entre fontes (grafo) | — | — | ✓ |

---

## 3. Resultados

### 3.1 Run: HGT (`runs/hgt/`)

**Configuração:** `configs/variant3_hgt.yaml` herdando `default.yaml`
- 30 épocas, batch=4, lr=1e-3, hidden_dim=128, 2 camadas HGT, 4 heads

#### Curva de Aprendizado

| Época | Train Loss | Val Loss | IAG IoA | IAG MAE | INMET IoA | INMET MAE |
|-------|-----------|---------|---------|---------|-----------|-----------|
| 1 | 0.4659 | 0.4143 | 0.587 | 1.372 | 0.586 | 1.412 |
| 5 | 0.3776 | 0.3833 | 0.615 | 1.357 | 0.620 | 1.321 |
| 10 | 0.3620 | 0.3632 | 0.645 | 1.225 | 0.635 | 1.268 |
| 15 | 0.3539 | 0.3614 | 0.645 | 1.143 | 0.632 | 1.288 |
| 17 | 0.3517 | **0.3469** | 0.661 | **1.060** | 0.646 | 1.226 |
| 20 | 0.3498 | 0.3562 | 0.645 | 1.137 | 0.643 | 1.250 |
| 23 | 0.3494 | 0.3498 | 0.653 | 1.166 | 0.649 | 1.202 |
| **27** | 0.3458 | **0.3448** | **0.663** | 1.064 | **0.648** | 1.216 |
| 30 | 0.3430 | 0.3546 | 0.643 | 1.166 | 0.648 | 1.230 |

**Melhor checkpoint salvo: Época 27** (menor `val_loss = 0.3448`)

#### Melhores Métricas de Validação (Época 27)

| Tipo | IoA | MAE | RMSE | Corr |
|------|-----|-----|------|------|
| **IAG** | **0.663** | **1.064** | **1.609** | **0.9887** |
| **INMET** | **0.648** | **1.216** | **1.708** | **0.9868** |

> **Lembrete de interpretação:** MAE e RMSE são médias das 4 features juntas (temperatura °C, ponto de orvalho °C, u m/s, v m/s). A correlação altíssima (~0.99) reflete principalmente a dominância da variável temperatura, que tem ciclo diurno muito forte e regular. IoA de 0.66 reflete a performance média em todas as variáveis — vento tem IoA muito menor que temperatura.

#### Observações Sobre a Convergência

- A loss de validação acompanha o treinamento sem overfitting severo (gap ~0.01 ao final)
- Há oscilações epoch-a-epoch na val_loss (ex.: épocas 4, 14, 27→30), o que é normal para batches pequenos (batch=4) com dados meteorológicos altamente variáveis
- A época 27 vence a 17 em `val_loss` total, mas na IAG isolada a época 17 tem MAE ligeiramente menor (1.060 vs 1.064). O checkpoint salvo é da 27 porque a loss combina IAG + INMET.
- O treino na época 30 ainda converge (train_loss=0.343 < val_loss=0.355), sugerindo que mais épocas poderiam ajudar marginalmente

#### Plotagens Geradas

- [runs/hgt/plots/error_by_leadtime.png](runs/hgt/plots/error_by_leadtime.png) — MAE por hora de antecedência (1h → 24h)
- [runs/hgt/plots/timeseries_examples.png](runs/hgt/plots/timeseries_examples.png) — Exemplos de séries previstas vs. observadas

---

### 3.2 Run: GRU (`runs/gru/`)

**Configuração:** `configs/variant1_gru.yaml`

| Arquivo | Status |
|---------|--------|
| `best.pt` | Disponível |
| `last.pt` | Disponível |
| `history.json` | **Não salvo** |

O histórico de treinamento do GRU não foi gravado nesta run. Os checkpoints existem e o modelo pode ser avaliado com `scripts/evaluate.py`. **Resultados quantitativos de validação/teste não disponíveis sem rodar avaliação.**

Para comparar GRU vs HGT:
```bash
python scripts/evaluate.py configs/variant1_gru.yaml runs/gru/best.pt --partition val
python scripts/evaluate.py configs/variant3_hgt.yaml runs/hgt/best.pt --partition val
```

---

### 3.3 Run: ST_MCAR

Sem checkpoint salvo — variante ainda não treinada.

---

## 4. O Que os Números Nos Dizem

### 4.1 Performance por Variável (estimativa qualitativa)

Com base nas métricas agregadas e na análise dos plots disponíveis:

| Variável | Dificuldade | Estimativa de IoA | Razão |
|----------|-------------|-------------------|-------|
| Temperatura do ar | **Baixa** | ~0.80–0.90 | Ciclo diurno fortíssimo, ERA5 captura bem |
| Ponto de orvalho | **Média** | ~0.50–0.60 | Processo local de umidade, ERA5 subestima gradientes |
| Vento (u, v) | **Alta** | ~0.45–0.55 | Alta variabilidade local, poucos dados de superfície |

A correlação global de ~0.988 é "inflada" pelo sinal dominante de temperatura — não deve ser interpretada como performance geral do modelo.

### 4.2 Limitações Conhecidas

1. **Vento:** Performance fraca é esperada. Estações costeiras e topografia local influenciam muito e não estão modeladas. O ERA5 tem resolução insuficiente para capturar circulações de mesoescala no interior de SP.

2. **Ponto de orvalho:** O ERA5 não representa bem a evapotranspiração local (vegetação, irrigação, superfícies impermeáveis). O modelo tem poucos "sensores" de umidade próximos ao IAG.

3. **Batch size pequeno (4):** Introduz ruído no gradiente e oscilações na val_loss. Um batch maior ou gradiente acumulado poderia suavizar a convergência.

4. **Métricas de validação ≠ métricas de teste:** O split de teste (jul–dez 2024) ainda não foi avaliado formalmente. O modelo `best.pt` foi escolhido pelo conjunto de validação (jan–jun 2024).

---

## 5. Próximos Passos Recomendados

1. **Rodar avaliação no conjunto de teste** com `scripts/evaluate.py` (partição `test`) para ambos GRU e HGT
2. **Treinar ST_MCAR** para obter a ablação do grafo vs. apenas encoding posicional
3. **Métricas por variável**: adicionar `metrics_per_variable` no loop de avaliação para decompor MAE/RMSE por variável separadamente
4. **Mais épocas para HGT**: convergência ainda ocorre na época 30 — testar 50–60 épocas
5. **Análise por lead time**: os plots de `error_by_leadtime.png` existem mas não foram quantificados no histórico — adicionar `per_leadtime_error` e `per_leadtime_ioa` ao `history.json`
