# Resource group + budget are ALWAYS applied (not behind enable_* flags).
# They notify; they do not freeze the subscription. Still destroy compute.

resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-rg"
  location = var.location

  tags = {
    Project   = "steam-recsys"
    Phase     = "6"
    ManagedBy = "terraform"
    BudgetCap = "10USD"
  }
}

resource "azurerm_monitor_action_group" "budget" {
  name                = "${var.project_name}-budget-alerts"
  resource_group_name = azurerm_resource_group.main.name
  short_name          = "recsbudget"

  email_receiver {
    name                    = "owner"
    email_address           = var.alert_email
    use_common_alert_schema = true
  }
}

resource "azurerm_consumption_budget_resource_group" "monthly" {
  name              = "${var.project_name}-monthly"
  resource_group_id = azurerm_resource_group.main.id
  amount            = var.budget_limit_usd
  time_grain        = "Monthly"

  time_period {
    start_date = "2026-09-01T00:00:00Z"
  }

  notification {
    enabled        = true
    threshold      = 50
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = [var.alert_email]
  }

  notification {
    enabled        = true
    threshold      = 80
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = [var.alert_email]
  }

  notification {
    enabled        = true
    threshold      = 100
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = [var.alert_email]
  }

  notification {
    enabled        = true
    threshold      = 100
    operator       = "GreaterThan"
    threshold_type = "Forecasted"
    contact_emails = [var.alert_email]
  }
}
