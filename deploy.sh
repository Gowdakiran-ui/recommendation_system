#!/bin/bash

# Neural Recommender System Kubernetes Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
NAMESPACE="neural-recommender"
DOCKER_IMAGE="neural-recommender:latest"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_prerequisites() {
    log_info "Checking prerequisites..."
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl required"; exit 1; }
    command -v docker >/dev/null 2>&1 || { log_error "docker required"; exit 1; }
    kubectl cluster-info >/dev/null 2>&1 || { log_error "Cannot connect to cluster"; exit 1; }
    log_success "Prerequisites OK"
}

build_docker_image() {
    log_info "Building Docker image..."
    [ ! -f "Dockerfile" ] && { log_error "Dockerfile not found"; exit 1; }
    docker build -t $DOCKER_IMAGE .
    log_success "Docker image built"
}

deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes..."
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -f k8s-deployment.yaml
    kubectl apply -f k8s-monitoring.yaml
    kubectl apply -f k8s-security.yaml
    kubectl apply -f k8s-jobs.yaml
    log_success "Deployment completed"
}

check_deployment() {
    log_info "Checking deployment..."
    kubectl get pods -n $NAMESPACE
    kubectl get services -n $NAMESPACE
    log_success "Deployment check completed"
}

# Main execution
log_info "🚀 Starting Neural Recommender deployment"
check_prerequisites
build_docker_image
deploy_to_kubernetes
check_deployment
log_success "🎉 Neural Recommender deployed successfully!"