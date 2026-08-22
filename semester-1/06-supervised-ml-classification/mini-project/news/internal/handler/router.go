package handler

import (
	"log/slog"
	"net/http"

	"github.com/riaz/newscollector/internal/observability"
)

// NewRouter wires the routes available in this milestone. Source, article and
// collection-run endpoints are added in later milestones.
func NewRouter(health *Health, logger *slog.Logger) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /health/live", health.Live)
	mux.HandleFunc("GET /health/ready", health.Ready)
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
