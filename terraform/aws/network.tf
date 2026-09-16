# Public subnet ONLY. No NAT Gateway. No private subnet.
# Instances get public IPs and reach ECR/S3 via IGW.

resource "aws_vpc" "main" {
  count = var.enable_network ? 1 : 0

  cidr_block           = "10.42.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  count = var.enable_network ? 1 : 0

  vpc_id = aws_vpc.main[0].id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

resource "aws_subnet" "public" {
  count = var.enable_network ? 1 : 0

  vpc_id                  = aws_vpc.main[0].id
  cidr_block              = "10.42.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = data.aws_availability_zones.available[0].names[0]

  tags = {
    Name = "${var.project_name}-public"
  }
}

resource "aws_route_table" "public" {
  count = var.enable_network ? 1 : 0

  vpc_id = aws_vpc.main[0].id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main[0].id
  }

  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  count = var.enable_network ? 1 : 0

  subnet_id      = aws_subnet.public[0].id
  route_table_id = aws_route_table.public[0].id
}

data "aws_availability_zones" "available" {
  count = var.enable_network ? 1 : 0
  state = "available"
}
