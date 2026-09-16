# Publish serving image to ECR and write the URI to SSM.
# Run from repo root after enable_network=true.
# Usage: .\scripts\publish_ecr.ps1 latest

param(
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$Ecr = terraform -chdir="terraform/aws" output -raw ecr_repository_url
$Param = terraform -chdir="terraform/aws" output -raw serving_image_param
$Region = "us-west-2"
if (-not $Ecr -or $Ecr -eq "null") {
    throw "ecr_repository_url is empty. Apply with enable_network=true first."
}
$Registry = $Ecr.Split("/")[0]

aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $Registry
docker build -f serving/Dockerfile -t "${Ecr}:${Tag}" .
docker push "${Ecr}:${Tag}"
aws ssm put-parameter --name $Param --type String --value "${Ecr}:${Tag}" --overwrite --region $Region
Write-Host "pushed ${Ecr}:${Tag}"
