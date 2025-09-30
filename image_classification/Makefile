.PHONY: build runb stopb endb runf stopf endf clean

build: clean
	docker-compose up --build

clean:
	# Remove all project containers using docker-compose
	docker-compose down --remove-orphans 2>/dev/null || true
	
	# Force remove any containers that might have these specific names
	-docker rm -f ml_service mon-front 2>/dev/null || true
	
	# Additional cleanup for any containers with service names
	-docker rm -f ml_service frontend 2>/dev/null || true
	
	# Clean up networks and volumes if needed
	-docker network prune -f 2>/dev/null || true

# backend
runb:
	docker-compose up ml_service

stopb:
	docker-compose stop ml_service

endb:
	docker-compose down ml_service

# frontend
runf:
	docker-compose up frontend

stopf:
	docker-compose stop frontend

endf:
	docker-compose down frontend
