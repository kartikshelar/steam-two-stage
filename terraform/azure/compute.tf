# CPU VM + Standard Load Balancer. Only when enable_compute=true.
# instance_count defaults to 0 — set to 1 only during a live timed session.
# No GPU. No NAT Gateway.

resource "azurerm_public_ip" "lb" {
  count = var.enable_compute ? 1 : 0

  name                = "${var.project_name}-lb-pip"
  location            = local.compute_location
  resource_group_name = azurerm_resource_group.main.name
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = { Name = "${var.project_name}-lb-pip" }
}

resource "azurerm_lb" "api" {
  count = var.enable_compute ? 1 : 0

  name                = "${var.project_name}-lb"
  location            = local.compute_location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "Standard"

  frontend_ip_configuration {
    name                 = "public"
    public_ip_address_id = azurerm_public_ip.lb[0].id
  }

  tags = { Name = "${var.project_name}-lb" }
}

resource "azurerm_lb_backend_address_pool" "cpu" {
  count = var.enable_compute ? 1 : 0

  name            = "${var.project_name}-bepool"
  loadbalancer_id = azurerm_lb.api[0].id
}

resource "azurerm_lb_probe" "health" {
  count = var.enable_compute ? 1 : 0

  name            = "http-health"
  loadbalancer_id = azurerm_lb.api[0].id
  protocol        = "Http"
  port            = 8000
  request_path    = "/health"
}

resource "azurerm_lb_rule" "http" {
  count = var.enable_compute ? 1 : 0

  name                           = "http-80"
  loadbalancer_id                = azurerm_lb.api[0].id
  protocol                       = "Tcp"
  frontend_port                  = 80
  backend_port                   = 8000
  frontend_ip_configuration_name = "public"
  backend_address_pool_ids       = [azurerm_lb_backend_address_pool.cpu[0].id]
  probe_id                       = azurerm_lb_probe.health[0].id
}

resource "azurerm_public_ip" "cpu" {
  count = var.enable_compute && var.instance_count == 1 ? 1 : 0

  name                = "${var.project_name}-cpu-pip"
  location            = local.compute_location
  resource_group_name = azurerm_resource_group.main.name
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = { Name = "${var.project_name}-cpu-pip" }
}

resource "azurerm_network_interface" "cpu" {
  count = var.enable_compute && var.instance_count == 1 ? 1 : 0

  name                = "${var.project_name}-cpu-nic"
  location            = local.compute_location
  resource_group_name = azurerm_resource_group.main.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.public[0].id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.cpu[0].id
  }
}

resource "azurerm_network_interface_backend_address_pool_association" "cpu" {
  count = var.enable_compute && var.instance_count == 1 ? 1 : 0

  network_interface_id    = azurerm_network_interface.cpu[0].id
  ip_configuration_name   = "internal"
  backend_address_pool_id = azurerm_lb_backend_address_pool.cpu[0].id
}

resource "azurerm_linux_virtual_machine" "cpu" {
  count = var.enable_compute && var.instance_count == 1 ? 1 : 0

  name                = "${var.project_name}-cpu"
  location            = local.compute_location
  resource_group_name = azurerm_resource_group.main.name
  size                = var.vm_size
  zone                = var.vm_zone != "" ? var.vm_zone : null
  admin_username      = var.admin_username
  network_interface_ids = [
    azurerm_network_interface.cpu[0].id,
  ]

  disable_password_authentication = true

  admin_ssh_key {
    username   = var.admin_username
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
    disk_size_gb         = 30
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }

  identity {
    type = "SystemAssigned"
  }

  custom_data = base64encode(templatefile("${path.module}/cloud_init_cpu.sh.tftpl", {
    acr_name        = azurerm_container_registry.serving[0].name
    acr_login       = azurerm_container_registry.serving[0].login_server
    storage_account = azurerm_storage_account.artifacts[0].name
    container       = azurerm_storage_container.bundle[0].name
  }))

  tags = { Name = "${var.project_name}-cpu" }

  lifecycle {
    precondition {
      condition     = var.allowed_cidr_api != ""
      error_message = "Set allowed_cidr_api to your public IP/32 before enabling compute."
    }
    precondition {
      condition     = var.ssh_public_key != ""
      error_message = "Set ssh_public_key before instance_count = 1."
    }
  }
}

resource "azurerm_role_assignment" "acr_pull" {
  count = var.enable_compute && var.instance_count == 1 ? 1 : 0

  scope                = azurerm_container_registry.serving[0].id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_linux_virtual_machine.cpu[0].identity[0].principal_id
}

resource "azurerm_role_assignment" "blob_read" {
  count = var.enable_compute && var.instance_count == 1 ? 1 : 0

  scope                = azurerm_storage_account.artifacts[0].id
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azurerm_linux_virtual_machine.cpu[0].identity[0].principal_id
}
