# Regional News Collection System — Phase 1

Collects articles from configurable RSS and Atom feeds into MongoDB.

Phase 1 is **data collection only**. Topic classification, bias analysis, location
matching, LLMs, site scraping, browser automation, social media and any frontend are
explicitly out of scope.

> **Status: Milestone 2 complete.** On top of Milestone 1's configuration,
> logging, MongoDB connection management, health endpoints, migration and
> container stack, the source model and its rules, the persistence layer, source
> management APIs and the seeding CLI are now in place. The feed collector, the
> processing pipeline, the scheduler and the article query APIs arrive in
> Milestones 3-6.

## Requirements

- Go 1.26+
- Docker and Docker Compose (for the local MongoDB and the container build)

## Quick start with Docker Compose

Starts MongoDB, runs the migration to completion, then starts the API:

```bash
make up
curl -s localhost:8080/health/live
curl -s localhost:8080/health/ready
make seed          # apply the feeds in configs/sources.yaml
make logs
make down          # stops the stack and deletes the volume
```

## Quick start without Docker

```bash
make mongo-up                      # MongoDB only, on localhost:27017
make migrate                       # create collections and indexes
make seed                          # apply the feeds in configs/sources.yaml
make run                           # serve on localhost:8080
```

## Verification gates

```bash
make check                         # gofmt check + go vet + go test ./...
make test-race                     # unit tests under the race detector
make cover                         # coverage summary
make test-integration              # needs MongoDB on localhost:27017
make seed-check                    # validate configs/sources.yaml, write nothing
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

## Sources

Feeds are declared in [configs/sources.yaml](configs/sources.yaml) and applied with
`make seed`. Seeding is idempotent and matches on `feed_url`, so re-running updates
rather than duplicates. An omitted optional key keeps whatever is already stored, so a
value tuned through the API is never reset by a re-seed. `make seed-check` validates the
whole file without writing, and a file containing an invalid entry is rejected in full
rather than applied halfway.

| Field                    | Required | Rule                                   |
| ------------------------ | -------- | -------------------------------------- |
| `name`                   | yes      | 1-200 characters                       |
| `feed_url`               | yes      | `http`/`https`, no credentials, unique |
| `type`                   | yes      | `rss` or `atom`                        |
| `language`               | yes      | two-letter ISO 639-1 code              |
| `country`                | yes      | two-letter ISO 3166-1 alpha-2 code     |
| `state`, `city`          | no       | up to 100 characters                   |
| `enabled`                | no       | defaults to `true`                     |
| `priority`               | no       | 0-100, defaults to 50                  |
| `fetch_interval_seconds` | no       | 60-604800, defaults to 900             |

`health_status`, `consecutive_failures`, `last_error`, `last_collected_at`,
`next_scheduled_at` and the timestamps are owned by the server. A request that tries to
set one is rejected rather than ignored.

## Endpoints

| Method   | Path                         | Purpose                                     | Milestone |
| -------- | ---------------------------- | ------------------------------------------- | --------- |
| `GET`    | `/health/live`               | Process is running; never touches MongoDB   | 1         |
| `GET`    | `/health/ready`              | `200` when MongoDB answers, `503` otherwise | 1         |
| `POST`   | `/api/v1/sources`            | Register a feed                             | 2         |
| `GET`    | `/api/v1/sources`            | List feeds, filtered and paginated          | 2         |
| `GET`    | `/api/v1/sources/{id}`       | Fetch one feed                              | 2         |
| `PATCH`  | `/api/v1/sources/{id}`       | Partially update a feed                     | 2         |
| `DELETE` | `/api/v1/sources/{id}`       | Remove a feed                               | 2         |
|          | `/api/v1/articles...`        | Article queries                             | 6         |
|          | `/api/v1/collection-runs...` | Run history                                 | 5         |

Liveness is deliberately independent of MongoDB so a transient database outage causes
a `503` on readiness rather than a restart loop.

Listing accepts `enabled`, `type`, `health_status`, `country`, `state`, `city`, `limit`
and `offset`. `limit` defaults to 50 and is capped at 100; asking for more is an error
rather than a silently truncated page, so nobody builds against a page size the server
will not honour.

```bash
curl -s 'localhost:8080/api/v1/sources?country=IN&enabled=true&limit=10'

curl -s -X POST localhost:8080/api/v1/sources \
  -H 'Content-Type: application/json' \
  -d '{"name":"Deccan Herald — Mysuru","feed_url":"https://www.deccanherald.com/rss/mysuru.xml",
       "type":"rss","language":"en","country":"IN","state":"Karnataka","city":"Mysuru"}'

curl -s -X PATCH localhost:8080/api/v1/sources/<id> \
  -H 'Content-Type: application/json' -d '{"enabled":false}'
```

Every error uses one envelope. Validation failures report every broken rule at once, so
one round trip is enough to fix a payload:

```json
{
  "error": {
    "code": "invalid_input",
    "message": "the request payload is invalid",
    "fields": [{ "field": "priority", "message": "must be between 0 and 100, got 500" }]
  }
}
```

| Code             | Status | Meaning                                 |
| ---------------- | ------ | --------------------------------------- |
| `invalid_input`  | 400    | The payload or a query parameter is bad |
| `not_found`      | 404    | No such source                          |
| `conflict`       | 409    | That feed URL is already registered     |
| `unavailable`    | 503    | A dependency did not answer in time     |
| `internal_error` | 500    | Anything else; detail stays in the logs |

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
cmd/seed          source seeding CLI
cmd/collector     manual collection CLI            (Milestone 5)

internal/config          layered configuration
internal/observability   slog setup, request ID, access log, panic recovery
internal/mongodb         connection management, collection names, index plan
internal/handler         thin HTTP handlers, request and response contracts
internal/domain          models and rules
internal/repository      persistence interfaces and storage-neutral errors
internal/repository/mongo MongoDB implementations
internal/service         application orchestration
internal/httpclient      SSRF-guarded HTTP client          (Milestone 3)
internal/collector/rss   gofeed-based RSS/Atom collector   (Milestone 3)
internal/processor       staged processing pipeline        (Milestone 4)
internal/scheduler       cron scheduling and locking       (Milestone 5)

configs/      default configuration and the source seed file
fixtures/     offline feed fixtures for tests    (Milestone 3)
deployments/  Dockerfile and Docker Compose
scripts/      developer helper scripts
```

Dependencies point inward only: handlers depend on services, services depend on
repository interfaces, and the MongoDB implementation is never imported by domain code.
Source identifiers are UUIDv7, so they are neither guessable nor enumerable, and because
they sort chronologically the `_id` index alone gives listings a stable order.

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
- Request bodies are capped at 64 KiB and must be `application/json`; unknown fields are
  rejected, so a caller cannot set a server-owned field such as `health_status`.
- Query filters are assembled field by field from typed, enum-validated values, so no
  part of a request is ever spliced into a query document as an operator.
- Feed URLs must be `http` or `https` and must not embed credentials, so a stored feed
  cannot leak a password into a log line.
- Source identifiers are UUIDv7 rather than sequential or timestamp-derived values.

The SSRF guard for user-supplied feed URLs arrives with the HTTP client in Milestone 3.
Until then the URL rules above reject only what can be judged without a DNS lookup: a
feed pointing at a private address is still accepted, because nothing fetches it yet.

**The source management API is unauthenticated.** Phase 1 has no identity layer, so
anyone who can reach the port can add or remove feeds. Do not expose it beyond a trusted
network.
