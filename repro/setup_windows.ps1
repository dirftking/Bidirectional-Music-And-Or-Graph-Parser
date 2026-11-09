param(
    [string]$MossEnvName = "moss-music-repro"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PIP_CONFIG_FILE = "NUL"
$env:PIP_EXTRA_INDEX_URL = "https://pypi.org/simple"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found on PATH"
}

$existingEnvs = conda env list --json | ConvertFrom-Json
$targetExists = $existingEnvs.envs | Where-Object { (Split-Path $_ -Leaf) -eq $MossEnvName }
if (-not $targetExists) {
    conda create -n $MossEnvName --override-channels -c conda-forge python=3.12 "ffmpeg=7" pip -y
}

conda run -n $MossEnvName python -m pip --isolated install --index-url https://download.pytorch.org/whl/cu128 --no-deps "torch==2.9.1+cu128" "torchaudio==2.9.1+cu128"
conda run -n $MossEnvName python -m pip install --index-url https://pypi.org/simple -c repro/constraints.txt -e ".[torch-runtime]"
conda run -n $MossEnvName python repro/check_environment.py
