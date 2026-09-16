# Terraform (Phase 5)

**Hard ceiling: $10 total. Alarm: $8. Public subnet only. No NAT. CPU, not GPU. Destroy every session.**

Read [`docs/aws-budget-guardrails.md`](../../docs/aws-budget-guardrails.md) before anything else.

This stack reuses the `llm-serving-bench` pattern (budget first, `enable_network` / `enable_compute` flags, public VPC, ALB, ASG, ECR, S3). Differences:

- Amazon Linux 2023 + `t3.medium` instead of a GPU DLAMI
- No GitHub OIDC provider (this account already has one from `llm-serving-bench`)
- 30 GB gp3 root volume, not 150 GB
- Serving image is CPU PyTorch; models/PIT bundle come from S3

## Safety defaults

| Flag | Default | Effect |
|---|---|---|
| `enable_network` | `false` | No VPC |
| `enable_compute` | `false` | No EC2 / ALB / ASG |
| `asg_max_size` | `1` | Cannot scale past 1 |
| `budget_limit_usd` | `8` (max 10) | Cannot set alarm above $10 |

ASG `desired_capacity` starts at **0** even when compute is enabled.

## First apply (budget only)

```powershell
cd terraform\aws
copy terraform.tfvars.example terraform.tfvars
# edit alert_email=
terraform init
terraform plan
terraform apply
```

Then confirm the **SNS subscription email** and that Billing → Budgets shows the steam-recsys budget.

## Later (short live session only)

1. Set `enable_network=true` (still no EC2)
2. Apply, push the serving image to ECR, sync `serving/bundle` to the artifacts bucket
3. Set `enable_compute=true` + `allowed_cidr_api="YOUR.IP/32"`
4. Apply, scale ASG to 1, measure, **immediately**:
   ```powershell
   terraform destroy
   ```
5. Console-check: EC2=0, ALB=0, NAT=0, unattached EBS=0, Elastic IPs=0

## Forbidden

- NAT Gateway
- GPU instances
- A second `aws_iam_openid_connect_provider` for GitHub
- `asg_max_size > 1`
- Leaving the ALB up overnight
