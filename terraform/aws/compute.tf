# CPU + ALB. Only when enable_compute=true.
# ASG desired_capacity = 0 by default — scale to 1 only during a live timed session.
# No GPU. No NAT. GitHub OIDC is not created here (this account already has one provider).

data "aws_ami" "al2023" {
  count = var.enable_compute ? 1 : 0

  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-kernel-6.1-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }
}

resource "aws_iam_role" "cpu" {
  count = var.enable_compute ? 1 : 0
  name  = "${var.project_name}-cpu-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "cpu" {
  count = var.enable_compute ? 1 : 0
  name  = "${var.project_name}-cpu-policy"
  role  = aws_iam_role.cpu[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket", "s3:PutObject"]
        Resource = [
          aws_s3_bucket.artifacts[0].arn,
          "${aws_s3_bucket.artifacts[0].arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters"
        ]
        Resource = "arn:aws:ssm:${var.aws_region}:*:parameter/${var.project_name}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["ec2messages:*", "ssm:UpdateInstanceInformation", "ssmmessages:*"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "cpu_ssm" {
  count = var.enable_compute ? 1 : 0

  role       = aws_iam_role.cpu[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "cpu" {
  count = var.enable_compute ? 1 : 0
  name  = "${var.project_name}-cpu-profile"
  role  = aws_iam_role.cpu[0].name
}

# ALB needs two AZs. Still public-only, no NAT.
resource "aws_subnet" "public_b" {
  count = var.enable_compute ? 1 : 0

  vpc_id                  = aws_vpc.main[0].id
  cidr_block              = "10.42.2.0/24"
  map_public_ip_on_launch = true
  availability_zone       = data.aws_availability_zones.available[0].names[1]

  tags = { Name = "${var.project_name}-public-b" }
}

resource "aws_route_table_association" "public_b" {
  count = var.enable_compute ? 1 : 0

  subnet_id      = aws_subnet.public_b[0].id
  route_table_id = aws_route_table.public[0].id
}

resource "aws_launch_template" "cpu" {
  count = var.enable_compute ? 1 : 0

  name_prefix   = "${var.project_name}-cpu-"
  image_id      = data.aws_ami.al2023[0].id
  instance_type = var.instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.cpu[0].name
  }

  vpc_security_group_ids = [aws_security_group.cpu[0].id]

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 30
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  user_data = base64encode(templatefile("${path.module}/user_data_cpu.sh.tftpl", {
    aws_region     = var.aws_region
    project_name   = var.project_name
    ecr_registry   = split("/", aws_ecr_repository.serving[0].repository_url)[0]
    ecr_repository = aws_ecr_repository.serving[0].repository_url
    image_param    = aws_ssm_parameter.serving_image[0].name
    s3_bucket      = aws_s3_bucket.artifacts[0].bucket
  }))

  tag_specifications {
    resource_type = "instance"
    tags          = { Name = "${var.project_name}-cpu" }
  }

  lifecycle {
    precondition {
      condition     = var.allowed_cidr_api != ""
      error_message = "Set allowed_cidr_api to your public IP/32 before enabling compute."
    }
  }
}

resource "aws_lb_target_group" "cpu" {
  count = var.enable_compute ? 1 : 0

  name     = "${var.project_name}-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main[0].id

  health_check {
    path                = "/health"
    matcher             = "200-399"
    interval            = 30
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }
}

resource "aws_lb" "api" {
  count = var.enable_compute ? 1 : 0

  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb[0].id]
  subnets            = [aws_subnet.public[0].id, aws_subnet.public_b[0].id]

  enable_deletion_protection = false

  tags = { Name = "${var.project_name}-alb" }
}

resource "aws_lb_listener" "http" {
  count = var.enable_compute ? 1 : 0

  load_balancer_arn = aws_lb.api[0].arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.cpu[0].arn
  }
}

resource "aws_autoscaling_group" "cpu" {
  count = var.enable_compute ? 1 : 0

  name                      = "${var.project_name}-cpu-asg"
  min_size                  = 0
  max_size                  = var.asg_max_size
  desired_capacity          = 0
  vpc_zone_identifier       = [aws_subnet.public[0].id, aws_subnet.public_b[0].id]
  health_check_type         = "EC2"
  health_check_grace_period = 420

  launch_template {
    id      = aws_launch_template.cpu[0].id
    version = "$Latest"
  }

  target_group_arns = [aws_lb_target_group.cpu[0].arn]

  tag {
    key                 = "Name"
    value               = "${var.project_name}-cpu"
    propagate_at_launch = true
  }

  lifecycle {
    ignore_changes = [desired_capacity]
  }
}

resource "aws_autoscaling_policy" "scale_to_one" {
  count = var.enable_compute ? 1 : 0

  name                   = "${var.project_name}-scale-to-one"
  autoscaling_group_name = aws_autoscaling_group.cpu[0].name
  adjustment_type        = "ExactCapacity"
  scaling_adjustment     = 1
  cooldown               = 60
}

resource "aws_cloudwatch_metric_alarm" "cpu_status_check" {
  count = var.enable_compute ? 1 : 0

  alarm_name          = "${var.project_name}-cpu-status-failed"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "StatusCheckFailed"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Maximum"
  threshold           = 0
  alarm_description   = "CPU instance status check failed"
  treat_missing_data  = "notBreaching"

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.cpu[0].name
  }

  alarm_actions = [aws_sns_topic.budget_alerts.arn]
}
