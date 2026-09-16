# Terraform Azure port (Phase 6)

This is a **port** of `terraform/aws`, not a second primary stack. AWS remains
the headline serving run (`results/serving_latency.csv`, hardware `aws_t3.medium`).
Azure exists so the same flags, budget-first order, public-only network, and CPU
serving VM can be stood up on a second provider.

Logged Azure loadgen: `20260916T085546Z`, hardware `azure_Standard_B2s_v2` in
`westus2`, p50/p95/p99 **129.8 / 174.5 / 228.2 ms**, 200 requests, concurrency 8,
errors=0. Source: `results/serving_latency.csv`. Not production traffic.

`Standard_B2s` had no capacity in `eastus` or `westus2` during that session.
ACR + storage stayed in `eastus`; VNet + VM used `compute_location = westus2`.

**Hard ceiling: $10. Alarm: $8. Public subnet only. No NAT Gateway. CPU, not GPU. Destroy every session.**

## Mapping (AWS → Azure)

| AWS | Azure |
|---|---|
| Budgets + SNS email | Cost Management budget on the resource group + action group email |
| VPC + public subnet + IGW | VNet `10.42.0.0/16` + subnet `10.42.1.0/24`, public IPs, **no NAT Gateway** |
| ECR | Azure Container Registry (Basic) |
| S3 artifacts bucket | Storage account + private `bundle` container |
| ALB :80 → :8000 `/health` | Standard Load Balancer, HTTP probe `/health` |
| ASG desired=0, max=1, `t3.medium` | `instance_count` 0 or 1, CPU B-series (`Standard_B2s` or `Standard_B2s_v2`) |
| Instance profile | System-assigned identity: AcrPull + Storage Blob Data Reader |
| GitHub OIDC | Not created (same reason as AWS Phase 5) |

## Safety defaults

| Flag | Default | Effect |
|---|---|---|
| `enable_network` | `false` | No VNet / ACR / storage |
| `enable_compute` | `false` | No VM / load balancer |
| `instance_count` | `0` | VM not created even if compute is enabled |
| `budget_limit_usd` | `8` (max 10) | Cannot set alarm above $10 |
| `compute_location` | empty | Uses `location`. Set to another region if B-series has no capacity |

The resource group and budget **always** apply. They are cheap. Confirm the
budget email before flipping `enable_network`.

Azure Linux VMs **reject ed25519** SSH keys. Use `ssh-rsa`. NSG rule priorities
must be 100–4096.

## Live session order

```powershell
az login
cd terraform\azure
copy terraform.tfvars.example terraform.tfvars
# edit alert_email=
terraform init
terraform plan
terraform apply
```

1. Confirm the budget email.
2. `enable_network=true`, apply, push the same `serving/Dockerfile` to ACR, upload `serving/bundle`.
3. `enable_compute=true` + `allowed_cidr_api` + RSA `ssh_public_key`, apply with `instance_count=0`.
4. Set `instance_count=1`, apply, loadgen against the LB public IP, **immediately**:
   ```powershell
   terraform destroy
   ```
5. Portal-check: 0 VMs, 0 load balancers, 0 NAT gateways, 0 public IPs.

If `Standard_B2s` returns `SkuNotAvailable`, set `compute_location` (e.g. `westus2`)
and/or `vm_size = "Standard_B2s_v2"`. Still CPU, still one VM.

## Forbidden

- Azure NAT Gateway
- GPU SKUs
- `instance_count > 1`
- Leaving the load balancer up overnight
- Reporting Azure p50/p95/p99 that are not in `results/serving_latency.csv`
