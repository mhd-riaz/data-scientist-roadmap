# Regional News Collection System — Phase 1

Collects articles from configurable RSS and Atom feeds into MongoDB.

Phase 1 is **data collection only**. Topic classification, bias analysis, location
matching, LLMs, site scraping, browser automation, social media and any frontend are
explicitly out of scope.

> **Status: Milestone 1 complete.** Configuration, structured logging, MongoDB
> connection management, health endpoints, the migration command and the container
> stack are in place. Source APIs, the collector, the processing pipeline, the
> scheduler and the article query APIs arrive in Milestones 2–6.

## Requirements

- Go 1.26+
- Docker and Docker Compose (for the local MongoDB and the container build)

## Quick start with Docker Compose

Starts MongoDB, runs the migration to completion, then starts the API:

```bash
make up
curl -s localhost:8080/health/live
curl -s localhost:8080/health/ready
make logs
make down          # stops the stack and deletes the volume
```

## Quick start without Docker

```bash
make mongo-up                      # MongoDB only, on localhost:27017
make migrate                       # create collections and indexes
make run                           # serve on localhost:8080
```

## Verification gates

```bash
make check                         # gofmt check + go vet + go test ./...
make test-race                     # unit tests under the race detector
make cover                         # coverage summary
make test-integration              # needs MongoDB on localhost:27017
make build                         # binaries into ./bin
```

Unit tests never contact a network service. The integration tests are behind the
`integration` build tag, use a database name unique per run, and drop it afterwards.

## Configuration

Three layers, each overriding the one before: built-in defaults →
[configs/config.yaml](configs/config.yaml) → environment variables. Every key maps to
`NEWS_<SECTION>_<KEY>`. Unknown YAML keys are rejected at startup so a typo fails fast.

| Key                              | Environment variable                  | Default                     |
| -------------------------------- | ------------------------------------- | --------------------------- |
| `app.name`                       | `NEWS_APP_NAME`                       | `news-collector`            |
| `app.environment`                | `NEWS_APP_ENVIRONMENT`                | `development`               |
| `server.host`                    | `NEWS_SERVER_HOST`                    | `0.0.0.0`                   |
| `server.port`                    | `NEWS_SERVER_PORT`                    | `8080`                      |
| `server.read_header_timeout`     | `NEWS_SERVER_READ_HEADER_TIMEOUT`     | `5s`                        |
| `server.read_timeout`            | `NEWS_SERVER_READ_TIMEOUT`            | `15s`                       |
| `server.write_timeout`           | `NEWS_SERVER_WRITE_TIMEOUT`           | `30s`                       |
| `server.idle_timeout`            | `NEWS_SERVER_IDLE_TIMEOUT`            | `60s`                       |
| `server.shutdown_timeout`        | `NEWS_SERVER_SHUTDOWN_TIMEOUT`        | `15s`                       |
| `server.max_header_bytes`        | `NEWS_SERVER_MAX_HEADER_BYTES`        | `1048576`                   |
| `mongo.uri`                      | `NEWS_MONGO_URI`                      | `mongodb://localhost:27017` |
| `mongo.database`                 | `NEWS_MONGO_DATABASE`                 | `news`                      |
| `mongo.connect_timeout`          | `NEWS_MONGO_CONNECT_TIMEOUT`          | `10s`                       |
| `mongo.server_selection_timeout` | `NEWS_MONGO_SERVER_SELECTION_TIMEOUT` | `10s`                       |
| `mongo.operation_timeout`        | `NEWS_MONGO_OPERATION_TIMEOUT`        | `30s`                       |
| `mongo.max_pool_size`            | `NEWS_MONGO_MAX_POOL_SIZE`            | `50`                        |
| `mongo.min_pool_size`            | `NEWS_MONGO_MIN_POOL_SIZE`            | `0`                         |
| `logging.level`                  | `NEWS_LOGGING_LEVEL`                  | `info`                      |
| `logging.format`                 | `NEWS_LOGGING_FORMAT`                 | `json`                      |

`NEWS_CONFIG_PATH` selects a different config file without passing `-config`.

## Endpoints

| Method | Path                         | Purpose                                     | Milestone |
| ------ | ---------------------------- | ------------------------------------------- | --------- |
| `GET`  | `/health/live`               | Process is running; never touches MongoDB   | 1         |
| `GET`  | `/health/ready`              | `200` when MongoDB answers, `503` otherwise | 1         |
|        | `/api/v1/sources...`         | Source management                           | 2         |
|        | `/api/v1/articles...`        | Article queries                             | 6         |
|        | `/api/v1/collection-runs...` | Run history                                 | 5         |

Liveness is deliberately independent of MongoDB so a transient database outage causes
a `503` on readiness rather than a restart loop.

## Data model

Five collections, created and indexed by `make migrate`:

| Collection            | Purpose                                                      |
| --------------------- | ------------------------------------------------------------ |
| `sources`             | Configured RSS/Atom feeds, their region, schedule and health |
| `articles`            | Normalized, deduplicated article metadata                    |
| `collection_runs`     | One audit record per collection attempt                      |
| `feed_cache_metadata` | ETag and Last-Modified per source, for HTTP 304              |
| `application_locks`   | TTL leases that stop two collectors sharing a source         |

The deduplication order (normalized URL → canonical URL → source + feed GUID → content
hash) is enforced by three unique indexes on `articles`. `content_hash` is indexed but
**not** unique, because two sources may legitimately syndicate identical content.
`application_locks` carries a TTL index so a crashed collector cannot hold a lock forever.

## Project layout

```text
cmd/api           HTTP server
cmd/migrate       collection and index migration
cmd/collector     manual collection CLI            (Milestone 5)
cmd/seed          source seeding CLI               (Milestone 2)

internal/config          layered configuration
internal/observability   slog setup, request ID, access log, panic recovery
internal/mongodb         connection management, collection names, index plan
internal/handler         thin HTTP handlers
internal/domain          models and rules                  (Milestone 2)
internal/repository      persistence interfaces            (Milestone 2)
internal/service         application orchestration         (Milestone 2)
internal/httpclient      SSRF-guarded HTTP client          (Milestone 3)
internal/collector/rss   gofeed-based RSS/Atom collector   (Milestone 3)
internal/processor       staged processing pipeline        (Milestone 4)
internal/scheduler       cron scheduling and locking       (Milestone 5)

configs/      default configuration
fixtures/     offline feed fixtures for tests    (Milestone 3)
deployments/  Dockerfile and Docker Compose
scripts/      developer helper scripts
```

Dependencies point inward only: handlers depend on services, services depend on
repository interfaces, and the MongoDB implementation is never imported by domain code.

## Security posture in this milestone

- MongoDB credentials are redacted before the connection target is logged.
- Request and response headers are never logged, so bearer tokens cannot leak.
- Driver errors are mapped to fixed codes; readiness reports `unavailable`, never the
  server-selection message naming internal hosts.
- Panics return an opaque `500` and are logged server-side without a stack in the body.
- Client-supplied `X-Request-Id` values are ignored; the server always generates its own.
- All JSON responses carry `X-Content-Type-Options: nosniff`.
- Configuration is validated at startup, including the MongoDB URI scheme and database name.
- The container image runs as UID 10001, never root.

The SSRF guard for user-supplied feed URLs arrives with the HTTP client in Milestone 3.
