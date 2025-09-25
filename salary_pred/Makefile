.PHONY: build runb stopb endb runf stopf endf

build:
	docker-compose up --build

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