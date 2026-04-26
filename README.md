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

## Notas de modelagem

- Vento é decomposto em `(u, v)` no data layer (config `data.decompose_wind`),
  para evitar perda circular sobre direção em graus.
- Tipos-alvo padrão: `IAG` e `INMET`. ERA5 entra apenas como contexto.
- Relações no grafo: `ERA5 → INMET`, `ERA5 → IAG`, `INMET → IAG`.
- A loss principal é `1 - IoA` (índice de concordância de Willmott).
