# Azure port (Phase 6)

AWS already served `POST /recommend` and was destroyed. This file is how to
claim the student Azure subscription and apply the **port** in `terraform/azure`.
It is not a second recsys stack.

Logged Azure loadgen: `20260916T085546Z`, hardware `azure_Standard_B2s_v2`,
p50/p95/p99 129.8 / 174.5 / 228.2 ms. Source: `results/serving_latency.csv`.
AWS `t3.medium` remains the headline serving number.

## Claim Azure for Students

1. Confirm GitHub Student Pack is active: https://education.github.com/pack
2. Open https://azure.microsoft.com/free/students/ (Azure for Students).
3. Sign in with the same Microsoft account you will use for `az login`.
4. Verify with GitHub when prompted.
5. Wait until the portal shows a subscription.
6. In Cloud Shell or local Azure CLI:
   ```powershell
   az login
   az account show
   ```
   If `az account show` fails, you do not have a subscription yet. Stop.
   Do not run Terraform.

Student credit and regional SKU availability change. Prefer `eastus`. If
`Standard_B2s` returns `SkuNotAvailable`, set `compute_location` (the 2026-09-16
session used `westus2`) or `vm_size = "Standard_B2s_v2"` — still CPU, still one
VM. Azure Linux VMs accept RSA SSH keys only (`ssh-rsa`), not ed25519.

## Budget first

Same order as AWS:

1. Copy `terraform/azure/terraform.tfvars.example` → `terraform.tfvars`.
2. Set `alert_email`. Leave `enable_network` and `enable_compute` false.
3. `terraform apply` (resource group + Cost Management budget only).
4. Confirm the budget email.
5. Then network, then a timed `instance_count=1` session, then
   `terraform destroy`.

Hard ceiling remains **$10**. Alarm **$8**. No NAT Gateway. Destroy every
session. Do not leave the load balancer up overnight.

## After a live session

Log p50/p95/p99 in `results/` with hardware labeled as the SKU that actually
ran (here: `azure_Standard_B2s_v2`). Until that file exists, Azure latency is
not a number this repo can report.

## What this port is not

Not production. Not a claim of Azure production recsys experience. Not a
reason to skip the AWS numbers already logged.
