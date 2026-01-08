# Terraform Infrastructure for SportAI

This directory contains Terraform configuration to deploy SportAI on AWS using ECS Fargate.

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform >= 1.0 installed
3. Appropriate AWS permissions for:
   - VPC creation
   - ECS cluster and service management
   - ECR repository creation
   - IAM role creation
   - CloudWatch log groups

## Setup

1. Copy the example variables file:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. Edit `terraform.tfvars` with your desired values

3. Initialize Terraform:
   ```bash
   terraform init
   ```

4. Review the plan:
   ```bash
   terraform plan
   ```

5. Apply the configuration:
   ```bash
   terraform apply
   ```

## Infrastructure Components

- **VPC**: Virtual Private Cloud with public and private subnets
- **ECS Cluster**: Fargate cluster for containerized application
- **ECR Repository**: Docker image repository
- **Application Load Balancer**: Public-facing load balancer
- **Security Groups**: Network security rules
- **CloudWatch**: Logging and monitoring

## Outputs

After applying, Terraform will output:
- ECR repository URL (for pushing Docker images)
- Load balancer DNS name (application URL)
- ECS cluster name
- VPC and security group IDs

## Deploying the Application

1. Build and push Docker image to ECR:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ECR_URL>
   docker build -t sportai .
   docker tag sportai:latest <ECR_URL>:latest
   docker push <ECR_URL>:latest
   ```

2. Update ECS service to use new image:
   ```bash
   aws ecs update-service --cluster sportai-cluster --service sportai-service --force-new-deployment
   ```

## Destroying Infrastructure

```bash
terraform destroy
```

## Notes

- The configuration uses Fargate for serverless container execution
- Health checks are configured for automatic recovery
- Logs are stored in CloudWatch with 7-day retention
- For production, enable deletion protection and configure HTTPS
