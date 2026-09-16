resource "aws_ecr_repository" "serving" {
  count = var.enable_network ? 1 : 0

  name                 = "${var.project_name}-serving"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }

  tags = {
    Name = "${var.project_name}-ecr"
  }
}

resource "aws_s3_bucket" "artifacts" {
  count = var.enable_network ? 1 : 0

  bucket_prefix = "${var.project_name}-artifacts-"
  force_destroy = true

  tags = {
    Name = "${var.project_name}-artifacts"
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  count = var.enable_network ? 1 : 0

  bucket = aws_s3_bucket.artifacts[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_ssm_parameter" "serving_image" {
  count = var.enable_network ? 1 : 0
  name  = "/${var.project_name}/serving_image"
  type  = "String"
  value = "placeholder:not-pushed-yet"

  lifecycle {
    ignore_changes = [value]
  }
}
