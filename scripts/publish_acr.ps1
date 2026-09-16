# Publish serving image to Azure Container Registry.
# Run from repo root after enable_network=true and `az login`.
# Usage: .\scripts\publish_acr.ps1 latest

param(
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$LoginServer = terraform -chdir="terraform/azure" output -raw acr_login_server
if (-not $LoginServer -or $LoginServer -eq "null") {
    throw "acr_login_server is empty. Apply with enable_network=true first."
}
$AcrName = $LoginServer.Split(".")[0]
$Image = "${LoginServer}/steam-recsys-serving:${Tag}"

az acr login --name $AcrName
docker build -f serving/Dockerfile -t $Image .
docker push $Image
Write-Host "pushed $Image"
