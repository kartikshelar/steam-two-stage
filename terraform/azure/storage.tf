resource "random_id" "suffix" {
  count       = var.enable_network ? 1 : 0
  byte_length = 3
}

resource "azurerm_container_registry" "serving" {
  count = var.enable_network ? 1 : 0

  name                = "${replace(var.project_name, "-", "")}${random_id.suffix[0].hex}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = false

  tags = {
    Name = "${var.project_name}-acr"
  }
}

resource "azurerm_storage_account" "artifacts" {
  count = var.enable_network ? 1 : 0

  name                     = "recs${random_id.suffix[0].hex}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"

  tags = {
    Name = "${var.project_name}-artifacts"
  }
}

resource "azurerm_storage_container" "bundle" {
  count = var.enable_network ? 1 : 0

  name                  = "bundle"
  storage_account_name  = azurerm_storage_account.artifacts[0].name
  container_access_type = "private"
}
