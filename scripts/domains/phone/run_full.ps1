Param(
  [string]$RunId = "$(Get-Date -Format 'yyyyMMdd')_phone",
  [string]$DataRootHint = ".\\data\\phone\\<brand>\\<model>\\*.json|jsonl"
)

$ErrorActionPreference = "Stop"
Write-Host "Ensure raw files exist under data/phone/<brand>/<model>/ (current hint: $DataRootHint)"

python -u .\scripts\pipeline_e2e.py `
  --domain phone `
  --run-id $RunId `
  --steps "00,tag,01,02,03,04,05,web"
