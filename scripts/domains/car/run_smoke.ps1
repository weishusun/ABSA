Param(
  [string]$InputAspectSentences = ".\outputs\car\aspect_sentences.parquet",
  [string]$RunId = "$(Get-Date -Format 'yyyyMMdd')_car_smoke"
)

$ErrorActionPreference = "Stop"

python -u .\scripts\route_b_sentiment\pipeline.py `
  --domain car `
  --run-id $RunId `
  --input-aspect-sentences $InputAspectSentences `
  --steps "01,02" `
  --max-train-rows 200 `
  --train-pool-rows 2000 `
  --shard-n 1 `
  --ds-batch-rows 1000 `
  --step02-max-rows 50
