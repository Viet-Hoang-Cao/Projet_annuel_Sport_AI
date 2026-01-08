output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.sportai.repository_url
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.sportai.dns_name
}

output "load_balancer_url" {
  description = "URL of the application"
  value       = "http://${aws_lb.sportai.dns_name}"
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "security_group_id" {
  description = "ID of the security group"
  value       = aws_security_group.sportai_sg.id
}
