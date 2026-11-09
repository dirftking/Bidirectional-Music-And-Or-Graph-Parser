param(
    [ValidateSet("Instruct", "Thinking")]
    [string]$Variant = "Instruct",
    [string]$MossEnvName = "moss-music-repro"
)

$ErrorActionPreference = "Stop"
$modelName = "MOSS-Music-8B-$Variant"
$target = Join-Path (Join-Path $PSScriptRoot "..\weights") $modelName

conda run -n $MossEnvName hf download "OpenMOSS-Team/$modelName" --local-dir $target
conda run -n $MossEnvName python repro/check_environment.py --model-path $target
