variable "location" {
  type        = string
  description = "Region for the resource group, ACR, and storage. eastus is the usual Azure for Students default."
  default     = "eastus"
}

variable "compute_location" {
  type        = string
  description = "Region for VNet + VM + load balancer. Empty uses location. Set when the primary region has no B-series capacity."
  default     = ""
}

variable "project_name" {
  type    = string
  default = "steam-recsys"
}

variable "alert_email" {
  type        = string
  description = "Email for Azure budget notifications. REQUIRED before apply."
}

variable "budget_limit_usd" {
  type        = number
  description = "Azure Cost Management alert threshold. Project ceiling is $10."
  default     = 8

  validation {
    condition     = var.budget_limit_usd <= 10
    error_message = "budget_limit_usd must be <= 10. Project ceiling is $10."
  }
}

variable "enable_network" {
  type        = bool
  description = "Create public-only VNet + ACR + storage. Default false until budget is confirmed."
  default     = false
}

variable "enable_compute" {
  type        = bool
  description = "Create CPU VM + load balancer. COSTS MONEY. Default false."
  default     = false

  validation {
    condition     = !(var.enable_compute && !var.enable_network)
    error_message = "enable_compute requires enable_network = true."
  }
}

variable "instance_count" {
  type        = number
  description = "0 or 1. Same role as AWS ASG desired_capacity. Default 0."
  default     = 0

  validation {
    condition     = contains([0, 1], var.instance_count)
    error_message = "instance_count must be 0 or 1 under the $10 ceiling."
  }
}

variable "vm_size" {
  type        = string
  description = "CPU SKU. Standard_B2s is 2 vCPU / 4 GiB, the t3.medium analogue. v2 B-series is allowed when v1 has no eastus capacity."
  default     = "Standard_B2s"

  validation {
    condition = contains([
      "Standard_B1s",
      "Standard_B2s",
      "Standard_B2ms",
      "Standard_B2s_v2",
      "Standard_B2ls_v2",
      "Standard_B2als_v2",
    ], var.vm_size)
    error_message = "vm_size must be a B-series CPU SKU. GPU is out of scope."
  }
}

variable "vm_zone" {
  type        = string
  description = "Optional availability zone (1, 2, or 3). Empty lets Azure pick a regional VM."
  default     = ""

  validation {
    condition     = contains(["", "1", "2", "3"], var.vm_zone)
    error_message = "vm_zone must be empty or 1, 2, or 3."
  }
}

variable "admin_username" {
  type    = string
  default = "azureuser"
}

variable "ssh_public_key" {
  type        = string
  description = "RSA SSH public key for the VM. Azure Linux VMs reject ed25519. Required when instance_count = 1."
  default     = ""

  validation {
    condition     = var.ssh_public_key == "" || startswith(var.ssh_public_key, "ssh-rsa ")
    error_message = "Azure Linux VMs only accept RSA SSH keys (ssh-rsa ...). ed25519 is rejected."
  }
}

variable "allowed_cidr_ssh" {
  type        = string
  description = "Optional SSH CIDR. Empty disables SSH ingress."
  default     = ""
}

variable "allowed_cidr_api" {
  type        = string
  description = "CIDR allowed to hit the load balancer (e.g. your IP/32). Required when compute is enabled."
  default     = ""
}
