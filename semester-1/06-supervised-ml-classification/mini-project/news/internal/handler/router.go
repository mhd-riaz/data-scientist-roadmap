package handler

import (
	"log/slog"
	"net/http"

	"github.com/riaz/newscollector/internal/observability"
)

// NewRouter wires the routes available in this milestone. Article and
// collection-run endpoints are added in later milestones.
func NewRouter(health *Health, sources *Source, logger *slog.Logger) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /health/live", health.Live)
	mux.HandleFunc("GET /health/ready", health.Ready)

	mux.HandleFunc("POST /api/v1/sources", sources.Create)
	mux.HandleFunc("GET /api/v1/sources", sources.List)
	mux.HandleFunc("GET /api/v1/sources/{id}", sources.Get)
	mux.HandleFunc("PATCH /api/v1/sources/{id}", sources.Update)
	mux.HandleFunc("DELETE /api/v1/sources/{id}", sources.Delete)

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
