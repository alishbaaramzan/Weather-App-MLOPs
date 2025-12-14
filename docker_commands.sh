#!/bin/bash
# Docker commands reference for Weather Prediction API

echo "==================================================================="
echo "Docker Commands for Weather Prediction API"
echo "==================================================================="
echo ""

# Build Docker image
echo "1. BUILD DOCKER IMAGE"
echo "-------------------------------------------------------------------"
echo "docker build -t weather-api:latest ."
echo ""

# Run container (single container)
echo "2. RUN CONTAINER (without docker-compose)"
echo "-------------------------------------------------------------------"
echo "docker run -d -p 8000:8000 --name weather-api weather-api:latest"
echo ""

# Run with docker-compose
echo "3. RUN WITH DOCKER COMPOSE (recommended)"
echo "-------------------------------------------------------------------"
echo "docker-compose up -d"
echo ""

# View logs
echo "4. VIEW LOGS"
echo "-------------------------------------------------------------------"
echo "docker logs weather-api"
echo "docker logs -f weather-api  # Follow logs"
echo "docker-compose logs -f      # With docker-compose"
echo ""

# Stop container
echo "5. STOP CONTAINER"
echo "-------------------------------------------------------------------"
echo "docker stop weather-api"
echo "docker-compose down"
echo ""

# Remove container
echo "6. REMOVE CONTAINER"
echo "-------------------------------------------------------------------"
echo "docker rm weather-api"
echo "docker-compose down -v  # Remove volumes too"
echo ""

# Check running containers
echo "7. CHECK RUNNING CONTAINERS"
echo "-------------------------------------------------------------------"
echo "docker ps"
echo "docker-compose ps"
echo ""

# Execute command inside container
echo "8. EXECUTE COMMANDS INSIDE CONTAINER"
echo "-------------------------------------------------------------------"
echo "docker exec -it weather-api bash"
echo "docker exec -it weather-api python -c 'from app.prediction import get_predictor; p = get_predictor(); print(p.get_model_info())'"
echo ""

# Rebuild image
echo "9. REBUILD IMAGE"
echo "-------------------------------------------------------------------"
echo "docker-compose build --no-cache"
echo "docker-compose up -d --build"
echo ""

# View container stats
echo "10. VIEW CONTAINER STATS"
echo "-------------------------------------------------------------------"
echo "docker stats weather-api"
echo ""

# Test API (from host machine)
echo "11. TEST API"
echo "-------------------------------------------------------------------"
echo "curl http://localhost:8000/"
echo "curl http://localhost:8000/health"
echo ""

# Push to Docker Hub (optional)
echo "12. PUSH TO DOCKER HUB (optional)"
echo "-------------------------------------------------------------------"
echo "docker tag weather-api:latest yourusername/weather-api:latest"
echo "docker push yourusername/weather-api:latest"
echo ""

echo "==================================================================="
echo "Quick Start:"
echo "  1. docker-compose up -d"
echo "  2. Open http://localhost:8000/docs"
echo "  3. docker-compose logs -f (to view logs)"
echo "==================================================================="