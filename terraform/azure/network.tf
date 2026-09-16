# Public subnet ONLY. No Azure NAT Gateway. VMs get public IPs and reach
# ACR / blob over the internet.

resource "azurerm_virtual_network" "main" {
  count = var.enable_network ? 1 : 0

  name                = "${var.project_name}-vnet"
  location            = local.compute_location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = ["10.42.0.0/16"]

  tags = {
    Name = "${var.project_name}-vnet"
  }
}

resource "azurerm_subnet" "public" {
  count = var.enable_network ? 1 : 0

  name                 = "${var.project_name}-public"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main[0].name
  address_prefixes     = ["10.42.1.0/24"]
}
