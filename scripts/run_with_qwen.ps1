param(
  [string]$CsvPath = "",
  [string]$CondaEnv = "mcrafter",
  [string]$ChatModel = "qwen3.7-plus",
  [string]$EmbeddingModel = "text-embedding-v4",
  [string]$BaseUrl = "",
  [int]$LlmTimeoutSeconds = 120,
  [int]$LlmMaxRetries = 2,
  [int]$LlmMaxTokens = 4096,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$PythonArgs
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

if (-not [string]::IsNullOrWhiteSpace($CsvPath) -and $CsvPath -notlike "*.csv") {
  $PythonArgs = @($CsvPath) + $PythonArgs
  $CsvPath = ""
}

if ([string]::IsNullOrWhiteSpace($CsvPath)) {
  $projectCsv = Join-Path $projectRoot "local_config\qwen_api_key.csv"
  if (Test-Path -LiteralPath $projectCsv) {
    $CsvPath = $projectCsv
  } else {
    $downloadDir = Join-Path $env:USERPROFILE "Downloads"
    $candidate = Get-ChildItem -LiteralPath $downloadDir -File -Filter "*apiKey-5685951.csv" |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1
    if ($candidate) {
      $CsvPath = $candidate.FullName
    }
  }
}

if (-not (Test-Path -LiteralPath $CsvPath)) {
  throw "Qwen CSV not found: $CsvPath"
}

$rows = Import-Csv -LiteralPath $CsvPath
if (-not $rows -or $rows.Count -eq 0) {
  throw "Qwen CSV is empty: $CsvPath"
}

$valueColumn = $rows[0].PSObject.Properties.Name | Where-Object { $_ -ne "id" } | Select-Object -First 1
if (-not $valueColumn) {
  throw "Could not find value column in Qwen CSV."
}

$config = @{}
foreach ($row in $rows) {
  $name = [string]$row.id
  if (-not [string]::IsNullOrWhiteSpace($name)) {
    $config[$name] = [string]$row.$valueColumn
  }
}

if ([string]::IsNullOrWhiteSpace($config["apiKey"])) {
  throw "Qwen CSV is missing apiKey."
}

if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
  $BaseUrl = $config["openAiCompatible"]
}
if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
  throw "Qwen CSV is missing openAiCompatible and no -BaseUrl was provided."
}

$env:LLM_PROVIDER = "qwen"
$env:QWEN_API_KEY = $config["apiKey"]
$env:QWEN_BASE_URL = $BaseUrl
$env:QWEN_CHAT_MODEL = $ChatModel
$env:QWEN_EMBEDDING_MODEL = $EmbeddingModel
$env:LLM_TIMEOUT_SECONDS = [string]$LlmTimeoutSeconds
$env:LLM_MAX_RETRIES = [string]$LlmMaxRetries
$env:LLM_MAX_TOKENS = [string]$LlmMaxTokens
$env:PYTHONUNBUFFERED = "1"

if (-not $PythonArgs -or $PythonArgs.Count -eq 0) {
  $PythonArgs = @("scripts\check_llm_config.py")
}

Write-Host "Using Qwen provider with chat model '$ChatModel' and embedding model '$EmbeddingModel'."
Write-Host "Base URL: $BaseUrl"

$condaCommand = Get-Command conda.exe -CommandType Application -ErrorAction SilentlyContinue |
  Select-Object -First 1
if ($condaCommand) {
  $condaExe = $condaCommand.Source
} else {
  $candidateCondaPaths = @(
    (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
    (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
    "C:\ProgramData\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\anaconda3\Scripts\conda.exe"
  )
  $condaExe = $candidateCondaPaths | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

if (-not $condaExe) {
  throw "Could not find conda.exe. Add conda to PATH or edit scripts\run_with_qwen.ps1 with your conda.exe path."
}

# Native stderr also carries warnings and progress bars. Preserve its text while
# using the process exit code to decide whether Python succeeded.
$previousErrorActionPreference = $ErrorActionPreference
try {
  $ErrorActionPreference = "Continue"
  & $condaExe run --no-capture-output --cwd $projectRoot -n $CondaEnv python @PythonArgs 2>&1 |
    ForEach-Object {
      if ($_ -is [System.Management.Automation.ErrorRecord]) {
        $_.Exception.Message
      } else {
        $_
      }
    }
  $pythonExitCode = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $previousErrorActionPreference
}
exit $pythonExitCode
