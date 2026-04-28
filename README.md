# meteorological-initial-model

Modelo de previsão meteorológica local (microclima USP) baseado em **Heterogeneous Graph
Transformer (HGT)**, integrando três fontes:

| Tipo | Papel | Estrutura |
|------|-------|-----------|
| `ERA5` | Contexto regional | grade regular (latitude × longitude) |
| `INMET` | Observações regionais complementares | múltiplas estações esparsas |
| `IAG` | Estação central da USP | uma instância principal |

A formulação é a do paper *Heterogeneous Graph Transformers for High-Resolution, Low-Cost
Metocean Forecasting and Monitoring* (OMAE2026-179637), adaptada para meteorologia
em superfície sobre a região da USP / SP.

## Estrutura

```
configs/                    YAML configs (default + por variante)
src/meteo_hgt/
  data/                     loaders do NetCDF unificado, janelas, MCAR, RevIN
  models/                   per-type encoder/decoder, codificações, HGT
  training/                 perdas (1 - IoA), métricas, trainer
  utils/                    seed, logging
scripts/                    train.py, evaluate.py, inspect_data.py
tests/                      smoke tests
data/                       (não versionado) meteo_unified.nc
```

## Dado de entrada

Espera-se um único arquivo NetCDF unificado em `data/meteo_unified.nc` no formato
descrito em [data/data_format.md](data/data_format.md):

- dim `instance` (~18.7k): cada estação ou ponto de grade é uma instância
- dim `time` (horária): timestamps em segundos UNIX
- variáveis por instância: `latitude`, `longitude`, `source_type` (`ERA5|IAG|INMET`),
  `instance_id`, `instance_name`
- variáveis (instance, time): `air_temperature_c`, `dew_point_temperature_c`,
  `wind_direction_deg`, `wind_speed_ms`

## Divisão treino / validação / teste

A divisão é **temporal (cronológica), não aleatória** — cada partição cobre
um intervalo de datas distinto, e os mesmos pontos no espaço (estações IAG,
INMET e pontos da grade ERA5) aparecem em todas elas. Só o eixo do tempo é
particionado.

Configurada em [configs/default.yaml](configs/default.yaml) (`data.splits`):

| Partição | Intervalo                    | Duração |
|----------|------------------------------|---------|
| Treino   | `2020-01-01` → `2022-12-31`  | 3 anos  |
| Validação| `2023-01-01` → `2023-12-31`  | 1 ano   |
| Teste    | `2024-01-01` → `2024-12-31`  | 1 ano   |

### Por que temporal (e não aleatório)

- Splits aleatórios em série temporal **vazam informação** entre treino e
  validação (autocorrelação horária / sazonal). Um split por ano garante
  que validação e teste contenham padrões sazonais nunca vistos no treino.
- Evita também que o mesmo dia, com observações próximas no tempo, apareça
  em duas partições diferentes.

### Como cada partição é convertida em amostras

Não treinamos sobre o ano inteiro de uma vez. Dentro de cada intervalo, o
loader enumera **"observer times"** `t_phi` numa grade fixa
([src/meteo_hgt/data/windows.py:27](src/meteo_hgt/data/windows.py#L27)):

- stride entre `t_phi`s: `windows.observer_stride_hours = 6 h`
- contexto: `[t_phi − 48 h, t_phi)` (`windows.context_hours = 48`)
- horizonte de previsão: `[t_phi, t_phi + 24 h]` (`windows.forecast_hours = 24`)

Um `t_phi` só é aceito se **as duas janelas (contexto + forecast) cabem
inteiramente dentro da partição e da cobertura do dataset**. Isso significa,
por exemplo, que para a partição de validação (2023):

- primeiro `t_phi` válido: `2023-01-03 00:00` (precisa de 48 h de contexto)
- último `t_phi` válido: `2023-12-30 00:00` (precisa de 24 h de forecast)

Como a grade tem stride fixa, **val e test são determinísticos**: a mesma
config produz exatamente o mesmo conjunto de janelas em qualquer máquina.

Número aproximado de janelas por partição (24 h/dia ÷ 6 h):

| Partição | Janelas por dia | Total aprox. |
|----------|-----------------|--------------|
| Treino   | 4               | ~4 380       |
| Validação| 4               | ~1 460       |
| Teste    | 4               | ~1 460       |

### Filtros espaciais aplicados antes da divisão temporal

Antes do split, `UnifiedStore` reduz o conjunto de instâncias
([src/meteo_hgt/runners.py:33](src/meteo_hgt/runners.py#L33)):

- ERA5: apenas pontos da grade dentro do bounding box
  `lat ∈ [−24.5, −23.0]`, `lon ∈ [−47.5, −46.0]`
  (`data.era5_region`).
- INMET: apenas estações dentro de `data.inmet_max_distance_km = 250 km`
  do centro IAG-USP (`data.iag_center = [-23.6510, -46.6225]`).
- IAG: estação central, mantida sempre.

Esses filtros são aplicados igualmente nas três partições — a divisão
treino/val/teste atua **apenas no eixo do tempo**.

### Tratamento de dados faltantes

NaNs no NetCDF (estação fora do ar, gap de horário etc.) são preservados
como uma máscara booleana
([src/meteo_hgt/data/dataset.py:123](src/meteo_hgt/data/dataset.py#L123)):
o tensor de feature recebe `0.0` no lugar do NaN, mas a máscara
`valid_ctx` / `valid_fcst` marca aquela posição como inválida. Loss e
métricas (`mae`, `rmse`, `corr`, `ioa`) ignoram entradas mascaradas. Isso
evita ter que descartar amostras só porque uma estação ficou silenciosa
por algumas horas.

## Variantes de modelo

Conforme §9 da especificação:

1. `gru` — GRU + RevIN (baseline mais simples)
2. `st_mcar` — GRU + codificação espaço-temporal + MCAR (sem HGT)
3. `hgt` — modelo completo com message passing heterogêneo

Selecionada via `model.variant` no config.

## Uso

Instalação:
```bash
pip install -e .
```

Treinamento:
```bash
python3 scripts/train.py --config configs/variant3_hgt.yaml
```

Avaliação em conjunto held-out:
```bash
python scripts/evaluate.py --config configs/variant3_hgt.yaml --checkpoint runs/.../best.pt
```

Inspeção rápida do NetCDF:
```bash
python scripts/inspect_data.py
```

Visualização dos resultados de uma run:
```bash
python3 scripts/plot_results.py \
  --config configs/variant3_hgt.yaml \
  --checkpoint runs/hgt/best.pt \
  --partition val
# escreve 4 PNGs em runs/hgt/plots/
```

## O que cada métrica significa

A loss usada no treino é `1 - IoA` (índice de concordância de Willmott). Para
cada (instância, variável) ela compara a previsão `ŷ` com o observado `y` ao
longo das 24 h do horizonte:

```
IoA = 1 - Σ (y - ŷ)²  /  Σ (|ŷ - ȳ| + |y - ȳ|)²
```

Como interpretar os números no log e nos gráficos:

| Métrica | Unidade | Como ler |
|---------|---------|----------|
| **IoA**  | adimensional, [0, 1] | 1 = previsão perfeita; ~0.6+ já é "razoável" para um modelo aprendendo do zero. **A loss = 1 - IoA**, então loss caindo de 0.66 → 0.30 = IoA subindo de 0.34 → 0.70. |
| **MAE**  | unidade da variável (°C, m/s) | Erro médio absoluto. Para `air_temperature_c`, MAE = 1.8 → o modelo erra ~1.8 °C em média. |
| **RMSE** | unidade da variável | Penaliza erros grandes mais que o MAE; sempre ≥ MAE. |
| **corr** | [-1, 1] | Correlação de Pearson. Diz se a forma da série está certa, ignorando viés/escala. Pode ser alta (0.95) com IoA baixo se houver viés. |

**Diagnóstico rápido**: se `corr` é alta mas `IoA` é baixa, o modelo está
acertando o padrão temporal mas com viés ou escala errada — geralmente
problema de normalização ou de dados de treino mal balanceados. Se ambos são
baixos, o sinal real ainda não está chegando aos pesos.

## Resultados desta run (variante `hgt`, val 2023)

Treinado por 30 epochs (`runs/hgt/`), com split temporal
train=2020–2022 / val=2023 / test=2024.

### Curva de aprendizado

A loss cai de forma estável e satura por volta do epoch 25–30 — não há
overfitting visível, val_loss continua acompanhando train_loss.

| Epoch | train_loss | val_loss |
|-------|-----------|----------|
| 1     | 0.554     | 0.435    |
| 10    | 0.427     | 0.379    |
| 20    | 0.413     | 0.376    |
| 30    | 0.408     | 0.371    |

Equivalente em IoA: o modelo sai de **IoA ≈ 0.57** (epoch 1) e termina em
**IoA ≈ 0.63–0.64** no melhor epoch.

### Métricas agregadas (melhor epoch)

| Tipo-alvo | IoA  | MAE  | RMSE | corr  |
|-----------|------|------|------|-------|
| **IAG**   | 0.64 | 1.17 | 1.75 | 0.985 |
| **INMET** | 0.62 | 1.23 | 1.84 | 0.983 |

(MAE/RMSE em unidades mistas — média sobre 4 features: °C, °C, m/s, m/s.)

### Por variável (held-out)

Quando se desagrega por variável (gráfico `plots/metrics_by_variable.png`), o
quadro fica bem desigual:

| Variável                | MAE        | RMSE       | IoA        | corr       | Veredito |
|-------------------------|------------|------------|------------|------------|----------|
| **Temperatura do ar**   | ~1.7 °C    | ~2.3–2.5 °C| 0.79–0.86  | ~0.87      | **bom**  |
| **Ponto de orvalho**    | ~1.3 °C    | ~1.8–2.0 °C| 0.52–0.56  | ~0.85      | razoável |
| **Vento u**             | ~0.8 m/s   | ~1.1–1.3 m/s| 0.52–0.58 | 0.51–0.58 | fraco    |
| **Vento v**             | ~0.85 m/s  | ~1.1–1.3 m/s| 0.53–0.66 | 0.53–0.58 | fraco    |

Leitura:
- **Temperatura do ar** é onde o modelo está claramente preciso: IoA alta
  (~0.8) *e* correlação alta (~0.87). O sinal sazonal/diurno está sendo
  capturado e o viés é pequeno.
- **Ponto de orvalho** tem correlação alta (~0.85) mas IoA baixa (~0.55).
  Diagnóstico: o modelo pega a *forma* da série, mas erra escala/viés —
  típico de variável com forte componente local que o ERA5 não resolve.
- **Vento (u, v)** está fraco em correlação (~0.55) — perto do que se
  esperaria de uma persistência climatológica. Vento é o sinal mais ruidoso
  e mais local; com apenas 5 anos e poucas estações, o modelo não está
  recuperando muita informação além da média.

### Erro vs. horizonte de previsão

Em `plots/error_by_leadtime.png`, o MAE cresce monotonicamente com o lead
time — comportamento esperado:

| Variável            | +1 h     | +24 h    | Crescimento |
|---------------------|----------|----------|-------------|
| Temperatura         | ~1.4 °C  | ~2.0 °C  | +0.6 °C     |
| Ponto de orvalho    | ~1.0 °C  | ~1.6 °C  | +0.6 °C     |
| Vento u/v           | ~0.7 m/s | ~1.0 m/s | +0.3 m/s    |

Ou seja: o modelo é mais útil nas **primeiras ~6–12 horas**; a partir daí o
erro se aproxima do erro climatológico.

## Para que aplicações este modelo está preciso?

Com base nos números acima, dá pra ser bem específico:

### Aplicações onde o modelo já é utilizável

1. **Previsão de temperatura do ar para curto prazo (1–12 h) na USP e
   entorno.** MAE ≈ 1.5–1.8 °C com correlação ~0.87 é suficiente para:
   - acompanhamento de microclima e suporte didático;
   - previsão de tendência (sobe / cai / estável) para conforto térmico;
   - input para modelos de demanda de energia/ventilação onde o erro de
     ~1.5 °C é tolerável.
2. **Previsão de tendência de ponto de orvalho.** A correlação alta (~0.85)
   permite usar o sinal qualitativamente (umidade subindo/descendo), mesmo
   com viés absoluto.
3. **Baseline e prova de conceito de fusão ERA5 + INMET + IAG num único
   modelo.** Os números mostram que o pipeline aprende — não é ruído.

### Aplicações onde o modelo **não** está preciso o suficiente

- **Previsão de vento operacional** (aviação, energia eólica, dispersão de
  poluentes): correlação ~0.55 é baixa demais. Precisa de mais dados,
  features adicionais (pressão, radiação) ou um regressor específico para
  vento.
- **Previsão de temperatura com tolerância < 1 °C** (agricultura de
  precisão, alerta de geada, conforto sensível): o MAE de ~1.7 °C é muito
  alto para esse caso de uso.
- **Forecast além de ~12 h** com qualidade estável: o erro cresce
  linearmente e em +24 h já está próximo do clima.
- **Substituir GFS/ECMWF**: este modelo ainda não foi comparado contra o
  baseline do §10.2 do spec; até essa comparação ser feita, ele é uma
  prova de conceito, não um substituto.

### Resumo em uma frase

> O modelo é **preciso para previsão de temperatura do ar de curto prazo
> (1–12 h) sobre a região USP/SP**, razoável para tendências de umidade, e
> ainda fraco para vento. Serve como baseline e plataforma para iterar,
> não como forecast operacional.

A escala temporal pequena (2020-2024) e o número limitado de estações fazem
desta uma versão "inicial" — análoga à do paper original, que usou 4 meses de
drifters como prova de conceito antes de escalar.

## Notas de modelagem

- Vento é decomposto em `(u, v)` no data layer (config `data.decompose_wind`),
  para evitar perda circular sobre direção em graus.
- Tipos-alvo padrão: `IAG` e `INMET`. ERA5 entra apenas como contexto.
- Relações no grafo: `ERA5 → INMET`, `ERA5 → IAG`, `INMET → IAG`.
- A loss principal é `1 - IoA` (índice de concordância de Willmott).
