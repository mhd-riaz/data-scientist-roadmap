BIN_DIR   := bin
VERSION   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
LDFLAGS   := -s -w -X main.version=$(VERSION)
COMPOSE   := docker compose -f deployments/docker-compose.yml
CONFIG    ?= configs/config.yaml
SOURCES   ?= configs/sources.yaml

.DEFAULT_GOAL := help

.PHONY: help
help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: tidy
tidy: ## Sync go.mod and go.sum
	go mod tidy

.PHONY: fmt
fmt: ## Format all Go sources
	gofmt -w .

.PHONY: fmt-check
fmt-check: ## Fail if any file is unformatted
	@unformatted=$$(gofmt -l .); \
	if [ -n "$$unformatted" ]; then echo "unformatted files:"; echo "$$unformatted"; exit 1; fi

.PHONY: vet
vet: ## Run go vet
	go vet ./...

.PHONY: build
build: ## Build every command into $(BIN_DIR)
	go build -trimpath -ldflags="$(LDFLAGS)" -o $(BIN_DIR)/api ./cmd/api
	go build -trimpath -ldflags="$(LDFLAGS)" -o $(BIN_DIR)/migrate ./cmd/migrate
	go build -trimpath -ldflags="$(LDFLAGS)" -o $(BIN_DIR)/seed ./cmd/seed
	go build -trimpath -ldflags="$(LDFLAGS)" -o $(BIN_DIR)/collector ./cmd/collector

.PHONY: test
test: ## Run unit tests (no external dependencies)
	go test ./...

.PHONY: test-race
test-race: ## Run unit tests with the race detector
	go test -race ./...

.PHONY: test-integration
test-integration: ## Run integration tests (requires MongoDB on localhost:27017)
	go test -tags=integration -count=1 ./...

.PHONY: cover
cover: ## Report unit-test coverage per package
	go test -coverprofile=coverage.out ./...
	go tool cover -func=coverage.out | tail -1

.PHONY: check
check: fmt-check vet test ## Run every gate required before a milestone is done

.PHONY: run
run: ## Run the API against the local config
	go run ./cmd/api -config $(CONFIG)

.PHONY: migrate
migrate: ## Create MongoDB collections and indexes
	go run ./cmd/migrate -config $(CONFIG)

.PHONY: seed
seed: ## Apply the sources declared in $(SOURCES)
	go run ./cmd/seed -config $(CONFIG) -sources $(SOURCES)

.PHONY: seed-check
seed-check: ## Validate the seed file without writing to the database
	go run ./cmd/seed -config $(CONFIG) -sources $(SOURCES) -dry-run

.PHONY: collect
collect: ## Collect every source that is currently due
	go run ./cmd/collector -config $(CONFIG)

.PHONY: collect-source
collect-source: ## Collect one source now: make collect-source SOURCE=<id>
	go run ./cmd/collector -config $(CONFIG) -source $(SOURCE)

.PHONY: up
up: ## Start MongoDB, run the migration and start the API
	$(COMPOSE) up --build -d

.PHONY: down
down: ## Stop the stack and remove its volumes
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Follow the API logs
	$(COMPOSE) logs -f api

.PHONY: mongo-up
mongo-up: ## Start only MongoDB, for local test runs
	$(COMPOSE) up -d mongo

.PHONY: clean
clean: ## Remove build and coverage artefacts
	rm -rf $(BIN_DIR) coverage.out coverage.html
