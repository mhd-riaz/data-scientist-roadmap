package handler

import (
	"log/slog"
	"net/http"

	"github.com/riaz/newscollector/internal/observability"
)

// NewRouter wires the routes available in this milestone. Article endpoints are
// added in a later milestone.
func NewRouter(health *Health, sources *Source, runs *CollectionRun, logger *slog.Logger) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /health/live", health.Live)
	mux.HandleFunc("GET /health/ready", health.Ready)

	mux.HandleFunc("POST /api/v1/sources", sources.Create)
	mux.HandleFunc("GET /api/v1/sources", sources.List)
	mux.HandleFunc("GET /api/v1/sources/{id}", sources.Get)
	mux.HandleFunc("PATCH /api/v1/sources/{id}", sources.Update)
	mux.HandleFunc("DELETE /api/v1/sources/{id}", sources.Delete)

	// Collections are driven by the scheduler and the collector CLI, never by a
	// caller, so the history is read-only over HTTP.
	mux.HandleFunc("GET /api/v1/collection-runs", runs.List)
	mux.HandleFunc("GET /api/v1/collection-runs/{id}", runs.Get)

	mux.HandleFunc("/", notFound)

	return observability.Chain(mux,
		observability.RequestID,
		observability.Recover(logger),
		observability.AccessLog(logger),
	)
}

func notFound(w http.ResponseWriter, _ *http.Request) {
	writeError(w, http.StatusNotFound, CodeNotFound, "the requested resource does not exist")
}
