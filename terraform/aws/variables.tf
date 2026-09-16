variable "aws_region" {
  type        = string
  description = "Region for Phase 5 resources. One region only."
  default     = "us-west-2"
}

variable "project_name" {
  type    = string
  default = "steam-recsys"
}

variable "alert_email" {
  type        = string
  description = "Email for AWS Budget and alarm notifications. REQUIRED before apply."
}

variable "budget_limit_usd" {
  type        = number
  description = "AWS Budgets alert threshold. Project ceiling is $10."
  default     = 8

  validation {
    condition     = var.budget_limit_usd <= 10
    error_message = "budget_limit_usd must be <= 10. Project ceiling is $10."
  }
}

variable "enable_network" {
  type        = bool
  description = "Create public-only VPC. Default false until budget is confirmed."
  default     = false
}

variable "enable_compute" {
  type        = bool
  description = "Create CPU + ALB + ASG. COSTS MONEY. Default false."
  default     = false

  validation {
    condition     = !(var.enable_compute && !var.enable_network)
    error_message = "enable_compute requires enable_network = true."
  }
}

variable "instance_type" {
  type        = string
  description = "CPU instance. t3.medium (4 GiB) is the default so the PIT freeze fits."
  default     = "t3.medium"

  validation {
    condition     = contains(["t3.small", "t3.medium", "t3.large"], var.instance_type)
    error_message = "instance_type must be a t3 CPU size. GPU is out of scope."
  }
}

variable "asg_max_size" {
  type        = number
  description = "Hard cap. Keep at 1 for budget safety."
  default     = 1

  validation {
    condition     = var.asg_max_size == 1
    error_message = "asg_max_size must be 1 under the $10 ceiling."
  }
}

variable "allowed_cidr_ssh" {
  type        = string
  description = "Optional SSH CIDR. Empty disables SSH ingress."
  default     = ""
}

variable "allowed_cidr_api" {
  type        = string
  description = "CIDR allowed to hit the ALB (e.g. your IP/32). Required when compute is enabled."
  default     = ""
}
