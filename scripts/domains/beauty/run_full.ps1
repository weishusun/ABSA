Param(
  [string]$RunId = "$(Get-Date -Format 'yyyyMMdd')_beauty",
  [string]$DataRootHint = ".\\data\\beauty\\<brand>\\<model>\\*.json|jsonl"
)

$ErrorActionPreference = "Stop"
Write-Host "Ensure raw files exist under data/beauty/<brand>/<model>/ (current hint: $DataRootHint)"

python -u .\scripts\pipeline_e2e.py `
  --domain beauty `
  --run-id $RunId `
  --steps "00,tag,01,02,03,04,05,web"
