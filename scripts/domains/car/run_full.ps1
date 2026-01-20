Param(
  [string]$RunId = "$(Get-Date -Format 'yyyyMMdd')_car",
  [string]$DataRootHint = ".\\data\\car\\<brand>\\<model>\\*.json|jsonl"
)

$ErrorActionPreference = "Stop"
Write-Host "Ensure raw files exist under data/car/<brand>/<model>/ (current hint: $DataRootHint)"

python -u .\scripts\pipeline_e2e.py `
  --domain car `
  --run-id $RunId `
  --steps "00,tag,01,02,03,04,05,web"
