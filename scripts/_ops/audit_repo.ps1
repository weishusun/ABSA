# scripts/_ops/audit_repo.ps1
$ErrorActionPreference = "Stop"

# 以仓库根目录为基准（脚本在 scripts/_ops 下，所以回到上两级）
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo

$auditDir = Join-Path $repo "docs\audit"
$cliDir   = Join-Path $auditDir "cli_help"
New-Item -ItemType Directory -Force -Path $cliDir | Out-Null

# 1) 基本信息：时间、路径、PowerShell、Python
$meta = @()
$meta += "time_utc:   $(Get-Date -Format u)"
$meta += "repo_root:  $repo"
$meta += "pwsh:       $($PSVersionTable.PSVersion)"
try { $meta += "python:     $(python --version 2>&1)" } catch { $meta += "python:     <not found>" }
try { $meta += "pip:        $(python -m pip --version 2>&1)" } catch { $meta += "pip:        <not found>" }
$meta | Set-Content (Join-Path $auditDir "meta.txt") -Encoding UTF8

# 2) Git 状态（若存在 git）
try {
  git rev-parse HEAD        | Set-Content (Join-Path $auditDir "git_head.txt") -Encoding UTF8
  git status --porcelain    | Set-Content (Join-Path $auditDir "git_status_porcelain.txt") -Encoding UTF8
  git log -1 --oneline      | Set-Content (Join-Path $auditDir "git_last_commit.txt") -Encoding UTF8
} catch {
  "git not available or not a repo" | Set-Content (Join-Path $auditDir "git_state_error.txt") -Encoding UTF8
}

# 3) 目录概览（顶层 + 关键目录大小）
Get-ChildItem -Force | Select-Object Mode,LastWriteTime,Length,Name |
  Format-Table -AutoSize | Out-String |
  Set-Content (Join-Path $auditDir "root_listing.txt") -Encoding UTF8

# 目录大小统计（只统计一层，避免很慢）
$dirs = @("scripts","configs","data","outputs","docs","review_pipeline","aspects",".venv")
$sizeLines = foreach ($d in $dirs) {
  $p = Join-Path $repo $d
  if (Test-Path $p) {
    $bytes = (Get-ChildItem $p -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    "{0,-16} {1,15:N0} bytes" -f $d, $bytes
  } else {
    "{0,-16} <missing>" -f $d
  }
}
$sizeLines | Set-Content (Join-Path $auditDir "dir_sizes.txt") -Encoding UTF8

# 4) 脚本入口清单（重点关注 scripts）
Get-ChildItem (Join-Path $repo "scripts") -Recurse -File -ErrorAction SilentlyContinue |
  Select-Object FullName,Length,LastWriteTime |
  Sort-Object FullName |
  Export-Csv (Join-Path $auditDir "scripts_inventory.csv") -NoTypeInformation -Encoding UTF8

# 5) 关键入口 --help（若文件存在则抓取）
function Dump-Help($path, $outName) {
  if (Test-Path $path) {
    $out = Join-Path $cliDir $outName
    python -u $path --help 2>&1 | Set-Content $out -Encoding UTF8
  }
}

Dump-Help ".\scripts\step00_ingest_json_to_clean_sentences.py" "step00_ingest_help.txt"
Dump-Help ".\scripts\tag_aspects.py" "tag_aspects_help.txt"
Dump-Help ".\scripts\route_b_sentiment\pipeline.py" "route_b_pipeline_help.txt"

# 6) configs 域包检查：四域 aspects.yaml 是否齐全
$domains = @("phone","car","laptop","beauty")
$cfgLines = foreach ($dom in $domains) {
  $p = Join-Path $repo ("configs\domains\{0}\aspects.yaml" -f $dom)
  if (Test-Path $p) { "OK  configs/domains/$dom/aspects.yaml" } else { "MISS configs/domains/$dom/aspects.yaml" }
}
$cfgLines | Set-Content (Join-Path $auditDir "domain_pack_check.txt") -Encoding UTF8

# 7) 搜索“绝对路径风险”（Windows盘符、/Users、/home 等）
$patterns = @(
  "([A-Za-z]:\\)",      # C:\ 这类
  "(/Users/)",          # mac 常见
  "(/home/)"            # linux 常见
)

$targets = @("scripts","review_pipeline","configs")
$hits = @()
foreach ($t in $targets) {
  $tp = Join-Path $repo $t
  if (Test-Path $tp) {
    $files = Get-ChildItem $tp -Recurse -File -ErrorAction SilentlyContinue
    foreach ($f in $files) {
      $content = Get-Content $f.FullName -ErrorAction SilentlyContinue
      foreach ($pat in $patterns) {
        if ($content -match $pat) {
          $hits += "$($f.FullName)  matches: $pat"
          break
        }
      }
    }
  }
}
if ($hits.Count -eq 0) { $hits = @("No obvious absolute-path patterns found in scripts/review_pipeline/configs") }
$hits | Set-Content (Join-Path $auditDir "absolute_path_scan.txt") -Encoding UTF8

# 8) 依赖快照（pip freeze）
try {
  python -m pip freeze 2>&1 | Set-Content (Join-Path $auditDir "pip_freeze.txt") -Encoding UTF8
} catch {
  "pip freeze failed" | Set-Content (Join-Path $auditDir "pip_freeze_error.txt") -Encoding UTF8
}

"OK. Audit written to: docs/audit" | Write-Host
