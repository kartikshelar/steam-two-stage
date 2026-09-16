output "resource_group" {
  value = azurerm_resource_group.main.name
}

output "budget_name" {
  value = azurerm_consumption_budget_resource_group.monthly.name
}

output "budget_limit_usd" {
  value = var.budget_limit_usd
}

output "hard_ceiling_usd" {
  value       = 10
  description = "Project hard stop. Not enforced automatically — you must destroy."
}

output "compute_location" {
  value = local.compute_location
}

output "enable_network" {
  value = var.enable_network
}

output "enable_compute" {
  value = var.enable_compute
}

output "instance_count" {
  value = var.instance_count
}

output "acr_login_server" {
  value = try(azurerm_container_registry.serving[0].login_server, null)
}

output "storage_account" {
  value = try(azurerm_storage_account.artifacts[0].name, null)
}

output "lb_public_ip" {
  value = try(azurerm_public_ip.lb[0].ip_address, null)
}

output "session_checklist" {
  value = <<-EOT
    BEFORE COMPUTE: confirm budget email + Cost Management alert on the resource group.
    LIVE SESSION: instance_count=1 only while measuring; never overnight.
    END SESSION: terraform destroy
    HARD CEILING: $10 total. Alarm at $8. Stop if email fires.
    No NAT Gateway. Public subnet only. CPU B-series, not GPU.
    This port has been applied and destroyed after the timed loadgen.
    Azure headline SKU for the logged row: Standard_B2s_v2 in westus2.
  EOT
}
