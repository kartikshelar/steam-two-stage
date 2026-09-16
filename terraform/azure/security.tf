resource "azurerm_network_security_group" "cpu" {
  count = var.enable_compute ? 1 : 0

  name                = "${var.project_name}-cpu-nsg"
  location            = local.compute_location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "lb-probe"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8000"
    source_address_prefix      = "AzureLoadBalancer"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "http-api"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = var.allowed_cidr_api != "" ? var.allowed_cidr_api : "127.0.0.1/32"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "app-direct"
    priority                   = 120
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8000"
    source_address_prefix      = var.allowed_cidr_api != "" ? var.allowed_cidr_api : "127.0.0.1/32"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "ssh"
    priority                   = 130
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = var.allowed_cidr_ssh != "" ? var.allowed_cidr_ssh : (var.allowed_cidr_api != "" ? var.allowed_cidr_api : "127.0.0.1/32")
    destination_address_prefix = "*"
  }

  tags = {
    Name = "${var.project_name}-cpu-nsg"
  }
}

resource "azurerm_subnet_network_security_group_association" "public" {
  count = var.enable_compute ? 1 : 0

  subnet_id                 = azurerm_subnet.public[0].id
  network_security_group_id = azurerm_network_security_group.cpu[0].id
}
