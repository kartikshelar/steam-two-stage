output "budget_name" {
  value = aws_budgets_budget.project_monthly.name
}

output "budget_limit_usd" {
  value = var.budget_limit_usd
}

output "hard_ceiling_usd" {
  value       = 10
  description = "Project hard stop. Not enforced by AWS automatically — you must destroy."
}

output "sns_budget_topic" {
  value = aws_sns_topic.budget_alerts.arn
}

output "enable_network" {
  value = var.enable_network
}

output "enable_compute" {
  value = var.enable_compute
}

output "vpc_id" {
  value = try(aws_vpc.main[0].id, null)
}

output "ecr_repository_url" {
  value = try(aws_ecr_repository.serving[0].repository_url, null)
}

output "s3_bucket" {
  value = try(aws_s3_bucket.artifacts[0].bucket, null)
}

output "serving_image_param" {
  value = try(aws_ssm_parameter.serving_image[0].name, null)
}

output "alb_dns_name" {
  value = try(aws_lb.api[0].dns_name, null)
}

output "asg_name" {
  value = try(aws_autoscaling_group.cpu[0].name, null)
}

output "session_checklist" {
  value = <<-EOT
    BEFORE COMPUTE: confirm SNS email subscription + Budgets alerts.
    LIVE SESSION: set desired=1 only while measuring; never overnight.
    END SESSION: terraform destroy
    HARD CEILING: $10 total. Alarm at $8. Stop if email fires.
    No NAT. Public subnet only. CPU t3, not GPU.
  EOT
}
