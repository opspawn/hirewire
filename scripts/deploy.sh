#!/usr/bin/env bash
# -------------------------------------------------------------------
# AgentOS — Azure Container Apps deployment script
#
# Usage:
#   ./scripts/deploy.sh          # build, push, deploy
#   ./scripts/deploy.sh build    # build only
#   ./scripts/deploy.sh push     # push to ACR only
#   ./scripts/deploy.sh deploy   # deploy to Container Apps only
# -------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

# Required variables
: "${ACR_LOGIN_SERVER:?ACR_LOGIN_SERVER not set}"
: "${ACR_USERNAME:?ACR_USERNAME not set}"
: "${ACR_PASSWORD:?ACR_PASSWORD not set}"
: "${CONTAINERAPP_ENV:?CONTAINERAPP_ENV not set}"
: "${AZURE_RESOURCE_GROUP:?AZURE_RESOURCE_GROUP not set}"

IMAGE_NAME="agentos-api"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"
APP_NAME="agentos-api"

# -------------------------------------------------------------------

build() {
    echo "==> Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "$PROJECT_DIR"
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "$FULL_IMAGE"
    echo "==> Build complete: $FULL_IMAGE"
}

push() {
    echo "==> Logging in to ACR: ${ACR_LOGIN_SERVER}"
    echo "$ACR_PASSWORD" | docker login "$ACR_LOGIN_SERVER" \
        --username "$ACR_USERNAME" --password-stdin

    echo "==> Pushing image: $FULL_IMAGE"
    docker push "$FULL_IMAGE"
    echo "==> Push complete"
}

deploy() {
    echo "==> Deploying to Azure Container Apps: ${APP_NAME}"

    # Check if the container app already exists
    if az containerapp show \
        --name "$APP_NAME" \
        --resource-group "$AZURE_RESOURCE_GROUP" \
        &>/dev/null; then
        echo "==> Updating existing container app"
        az containerapp update \
            --name "$APP_NAME" \
            --resource-group "$AZURE_RESOURCE_GROUP" \
            --image "$FULL_IMAGE" \
            --set-env-vars \
                "AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}" \
                "AZURE_OPENAI_KEY=${AZURE_OPENAI_KEY}" \
                "AZURE_OPENAI_DEPLOYMENT=${AZURE_OPENAI_DEPLOYMENT}" \
                "COSMOS_ENDPOINT=${COSMOS_ENDPOINT}" \
                "COSMOS_KEY=${COSMOS_KEY}" \
                "MODEL_PROVIDER=azure_ai" \
                "AGENTOS_DEMO=1" \
                "PYTHONUNBUFFERED=1"
    else
        echo "==> Creating new container app"
        az containerapp create \
            --name "$APP_NAME" \
            --resource-group "$AZURE_RESOURCE_GROUP" \
            --environment "$CONTAINERAPP_ENV" \
            --image "$FULL_IMAGE" \
            --registry-server "$ACR_LOGIN_SERVER" \
            --registry-username "$ACR_USERNAME" \
            --registry-password "$ACR_PASSWORD" \
            --target-port 8000 \
            --ingress external \
            --min-replicas 0 \
            --max-replicas 3 \
            --cpu 0.5 \
            --memory 1Gi \
            --env-vars \
                "AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}" \
                "AZURE_OPENAI_KEY=${AZURE_OPENAI_KEY}" \
                "AZURE_OPENAI_DEPLOYMENT=${AZURE_OPENAI_DEPLOYMENT}" \
                "COSMOS_ENDPOINT=${COSMOS_ENDPOINT}" \
                "COSMOS_KEY=${COSMOS_KEY}" \
                "MODEL_PROVIDER=azure_ai" \
                "AGENTOS_DEMO=1" \
                "PYTHONUNBUFFERED=1"
    fi

    # Get the FQDN
    FQDN=$(az containerapp show \
        --name "$APP_NAME" \
        --resource-group "$AZURE_RESOURCE_GROUP" \
        --query "properties.configuration.ingress.fqdn" \
        --output tsv)

    echo ""
    echo "========================================="
    echo "  AgentOS deployed successfully!"
    echo "  URL: https://${FQDN}"
    echo "  Health: https://${FQDN}/health"
    echo "  Azure: https://${FQDN}/health/azure"
    echo "  Dashboard: https://${FQDN}/dashboard"
    echo "  API Docs: https://${FQDN}/docs"
    echo "========================================="
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

case "${1:-all}" in
    build)  build ;;
    push)   push ;;
    deploy) deploy ;;
    all)    build && push && deploy ;;
    *)
        echo "Usage: $0 {build|push|deploy|all}"
        exit 1
        ;;
esac
