#!/bin/ksh
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""
export AWS_SESSION_TOKEN=""

set -e  # Exit immediately if a command exits with a non-zero status

AWS_PUBLIC_ECR="public.ecr.aws/s2e1n3u8"
AWS_REGION=$(aws configure get region)

aws ecr-public get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_PUBLIC_ECR

services=(
    "maap-meta-qs-main"
    "maap-meta-qs-ui"
    "maap-meta-qs-loader"
    "maap-meta-qs-logger"
    "maap-meta-qs-ai-memory"
    "maap-meta-qs-semantic-cache"
    "maap-meta-qs-nginx"
)

cd MAAP-AWS-Meta

# Ensure Docker Buildx is enabled for multi-platform builds
docker buildx create --use || echo "Buildx already enabled"

for service in "${services[@]}"; do
    echo "======================================"
    sub_service="${service#maap-meta-qs-}"
    cd "$sub_service"
    echo "🚀 Building and pushing multi-platform image for $service..."
    docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag "$AWS_PUBLIC_ECR/$service:latest" \
    --push .
    cd ..
    echo "======================================"
done

echo "🎉 All images pushed successfully!"
