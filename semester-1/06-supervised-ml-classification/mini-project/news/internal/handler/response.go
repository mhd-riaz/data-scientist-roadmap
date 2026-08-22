// Package handler holds the thin HTTP layer. Handlers bind and validate input,
// delegate to application services, and never expose internal error detail.
package handler

import (
	"encoding/json"
	"net/http"
)

// Error codes returned to clients. They are deliberately coarse so that no
// database, driver or topology detail reaches the caller.
const (
	CodeNotFound      = "not_found"
	CodeInternalError = "internal_error"
	CodeUnavailable   = "unavailable"
)

// errorEnvelope is the single error shape used by every endpoint.
type errorEnvelope struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// writeJSON serialises payload before touching the response, so an encoding
// failure cannot emit a half-written body under an already-sent 200.
func writeJSON(w http.ResponseWriter, status int, payload any) {
	body, err := json.Marshal(payload)
	if err != nil {
		writeError(w, http.StatusInternalServerError, CodeInternalError, "internal server error")
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

// writeError emits the error envelope. message must be a fixed, caller-safe
// string, never a wrapped internal error.
func writeError(w http.ResponseWriter, status int, code, message string) {
	body, err := json.Marshal(errorEnvelope{Error: errorDetail{Code: code, Message: message}})
	if err != nil {
		http.Error(w, "internal server error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}
