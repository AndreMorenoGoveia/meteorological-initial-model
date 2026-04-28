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

## Para que serve este modelo inicial?

Mais do que números absolutos, este projeto entrega um **pipeline e baseline de
referência**:

1. **Baseline contra GFS** (§10.2 do spec): permite medir se o HGT, treinado
   apenas com 5 anos de dados locais, supera ou empata um forecast operacional
   global em algumas variáveis na região da USP.
2. **Demonstração de fusão multi-fonte**: mostra que dá pra integrar reanálise
   em grade (ERA5) com observações irregulares (IAG, INMET) num modelo só, sem
   precisar interpolar tudo numa grade comum.
3. **Plataforma para iterar**: as três variantes (`gru` / `st_mcar` / `hgt`)
   permitem isolar o ganho real do grafo heterogêneo vs codificação ST vs apenas
   uma RNN. Os gráficos do `plot_results.py` deixam essa comparação visual.
4. **Forecast curto-prazo de microclima**: depois de treinado em dados reais,
   produz previsão horária para 24h à frente para a estação IAG e estações
   INMET próximas, condicionada ao contexto regional do ERA5.

A escala temporal pequena (2020-2024) e o número limitado de estações fazem
desta uma versão "inicial" — análoga à do paper original, que usou 4 meses de
drifters como prova de conceito antes de escalar.

## Notas de modelagem

- Vento é decomposto em `(u, v)` no data layer (config `data.decompose_wind`),
  para evitar perda circular sobre direção em graus.
- Tipos-alvo padrão: `IAG` e `INMET`. ERA5 entra apenas como contexto.
- Relações no grafo: `ERA5 → INMET`, `ERA5 → IAG`, `INMET → IAG`.
- A loss principal é `1 - IoA` (índice de concordância de Willmott).
