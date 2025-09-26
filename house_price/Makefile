.PHONY: build runb stopb endb runf stopf endf clean

build:
	docker-compose up --build

clean:
	docker-compose down ml_service
	docker-compose down frontend
	docker rm -f ml_service 2>/dev/null || true
	docker rm -f frontend 2>/dev/null || true

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
