# AWS budget guardrails (steam-recsys)

Hard ceiling: **$10** total for this project. Alarm at **$8**. CPU only. Public subnet only. No NAT Gateway. Destroy at the end of every session.

These rules are copied from `llm-serving-bench` and tightened for a cheaper CPU serving session. They are not optional.

## Before any `terraform apply`

1. Confirm the AWS identity is the student account used for `llm-serving-bench` (not a personal unused account, not a lab account you cannot destroy).
2. Copy `terraform/aws/terraform.tfvars.example` → `terraform/aws/terraform.tfvars`.
3. Set `alert_email` to an inbox you will actually check. Do not commit that file.
4. Leave `enable_network = false` and `enable_compute = false`.
5. `terraform apply` **budget + SNS only**.
6. Confirm the SNS subscription email.
7. Confirm Billing → Budgets shows `steam-recsys-monthly` at $8.

Do not create a VPC until that checklist is done.

## Network

Public subnet + internet gateway only. Instances get public IPs. No NAT Gateway, no private subnet, no VPC endpoints that bill by the hour.

## Compute

- `t3.medium` on-demand. GPU is out of scope (this is not `llm-serving-bench`).
- Spot is not required: a 30-minute `t3.medium` + ALB session is cents, and a spot kill mid-loadgen wastes the session.
- ASG `desired_capacity` starts at 0. Scale to 1 only while measuring.
- `asg_max_size` is locked at 1.
- Do not leave the ALB up overnight (~$16/month by itself).

## GitHub OIDC

Do **not** copy `github_oidc.tf`. An AWS account may have only one IAM OIDC provider for `token.actions.githubusercontent.com`. That provider already exists from `llm-serving-bench`. Phase 5 publishes the image from a laptop with existing AWS keys.

## End of session

```powershell
cd terraform\aws
terraform destroy
```

Then in the console: EC2 instances = 0, load balancers = 0, NAT gateways = 0, unattached EBS = 0, Elastic IPs = 0.

If a budget email fires, destroy immediately. The alarm notifies; it does not freeze the account.
