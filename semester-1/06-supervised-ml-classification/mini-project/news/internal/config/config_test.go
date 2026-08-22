package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

func TestDefaultIsValid(t *testing.T) {
	if err := Default().Validate(); err != nil {
		t.Fatalf("built-in defaults must validate, got: %v", err)
	}
}

func TestLoadWithoutFileUsesDefaults(t *testing.T) {
	cfg, err := Load("")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Server.Port != 8080 {
		t.Errorf("Server.Port = %d, want 8080", cfg.Server.Port)
	}
	if cfg.Mongo.Database != "news" {
		t.Errorf("Mongo.Database = %q, want %q", cfg.Mongo.Database, "news")
	}
}

func TestLoadFileOverridesDefaultsAndParsesDurations(t *testing.T) {
	path := writeConfig(t, `
app:
  name: custom-app
server:
  port: 9090
  shutdown_timeout: 45s
mongo:
  database: custom_db
logging:
  level: debug
  format: text
`)

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.App.Name != "custom-app" {
		t.Errorf("App.Name = %q, want %q", cfg.App.Name, "custom-app")
	}
	if cfg.Server.Port != 9090 {
		t.Errorf("Server.Port = %d, want 9090", cfg.Server.Port)
	}
	if cfg.Server.ShutdownTimeout != 45*time.Second {
		t.Errorf("Server.ShutdownTimeout = %s, want 45s", cfg.Server.ShutdownTimeout)
	}
	if cfg.Mongo.Database != "custom_db" {
		t.Errorf("Mongo.Database = %q, want %q", cfg.Mongo.Database, "custom_db")
	}
	if cfg.Server.ReadTimeout != 15*time.Second {
		t.Errorf("unspecified keys must keep their default, got ReadTimeout = %s", cfg.Server.ReadTimeout)
	}
}

func TestLoadRejectsUnknownKeys(t *testing.T) {
	path := writeConfig(t, "server:\n  prot: 9090\n")

	_, err := Load(path)
	if err == nil {
		t.Fatal("expected an error for a misspelled config key")
	}
	if !strings.Contains(err.Error(), "prot") {
		t.Errorf("error should name the unknown field, got: %v", err)
	}
}

func TestLoadReportsMissingFile(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "absent.yaml")); err == nil {
		t.Fatal("expected an error for a missing config file")
	}
}

func TestEnvOverridesFile(t *testing.T) {
	path := writeConfig(t, "server:\n  port: 9090\n")

	t.Setenv(EnvPrefix+"SERVER_PORT", "7070")
	t.Setenv(EnvPrefix+"MONGO_URI", "mongodb://mongo:27017")
	t.Setenv(EnvPrefix+"MONGO_MAX_POOL_SIZE", "25")
	t.Setenv(EnvPrefix+"SERVER_SHUTDOWN_TIMEOUT", "3s")

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Server.Port != 7070 {
		t.Errorf("Server.Port = %d, want the env value 7070", cfg.Server.Port)
	}
	if cfg.Mongo.URI != "mongodb://mongo:27017" {
		t.Errorf("Mongo.URI = %q", cfg.Mongo.URI)
	}
	if cfg.Mongo.MaxPoolSize != 25 {
		t.Errorf("Mongo.MaxPoolSize = %d, want 25", cfg.Mongo.MaxPoolSize)
	}
	if cfg.Server.ShutdownTimeout != 3*time.Second {
		t.Errorf("Server.ShutdownTimeout = %s, want 3s", cfg.Server.ShutdownTimeout)
	}
}

func TestEnvBlankValueIsIgnored(t *testing.T) {
	t.Setenv(EnvPrefix+"MONGO_DATABASE", "   ")

	cfg, err := Load("")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Mongo.Database != "news" {
		t.Errorf("blank env var must not clear the default, got %q", cfg.Mongo.Database)
	}
}

func TestEnvMalformedValuesAreReported(t *testing.T) {
	t.Setenv(EnvPrefix+"SERVER_PORT", "not-a-number")
	t.Setenv(EnvPrefix+"MONGO_CONNECT_TIMEOUT", "soon")

	_, err := Load("")
	if err == nil {
		t.Fatal("expected an error for malformed environment values")
	}
	for _, want := range []string{"SERVER_PORT", "MONGO_CONNECT_TIMEOUT"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error should mention %s, got: %v", want, err)
		}
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantMsg string
	}{
		{"port too low", func(c *Config) { c.Server.Port = 0 }, "server.port"},
		{"port too high", func(c *Config) { c.Server.Port = 70000 }, "server.port"},
		{"empty app name", func(c *Config) { c.App.Name = "" }, "app.name"},
		{"unknown environment", func(c *Config) { c.App.Environment = "qa" }, "app.environment"},
		{"non-positive timeout", func(c *Config) { c.Server.ShutdownTimeout = 0 }, "server.shutdown_timeout"},
		{"empty mongo uri", func(c *Config) { c.Mongo.URI = "" }, "mongo.uri"},
		{"wrong mongo scheme", func(c *Config) { c.Mongo.URI = "http://localhost:27017" }, "mongo.uri"},
		{"empty database", func(c *Config) { c.Mongo.Database = "" }, "mongo.database"},
		{"illegal database char", func(c *Config) { c.Mongo.Database = "news/db" }, "mongo.database"},
		{"zero pool", func(c *Config) { c.Mongo.MaxPoolSize = 0 }, "mongo.max_pool_size"},
		{"min above max", func(c *Config) { c.Mongo.MinPoolSize = 100; c.Mongo.MaxPoolSize = 10 }, "mongo.min_pool_size"},
		{"unknown log level", func(c *Config) { c.Logging.Level = "verbose" }, "logging.level"},
		{"unknown log format", func(c *Config) { c.Logging.Format = "xml" }, "logging.format"},
		{"empty user agent", func(c *Config) { c.Collector.UserAgent = " " }, "collector.user_agent"},
		{"zero request timeout", func(c *Config) { c.Collector.RequestTimeout = 0 }, "collector.request_timeout"},
		{"zero response cap", func(c *Config) { c.Collector.MaxResponseBytes = 0 }, "collector.max_response_bytes"},
		{"response cap too high", func(c *Config) { c.Collector.MaxResponseBytes = 1 << 40 }, "collector.max_response_bytes"},
		{"negative redirects", func(c *Config) { c.Collector.MaxRedirects = -1 }, "collector.max_redirects"},
		{"too many redirects", func(c *Config) { c.Collector.MaxRedirects = 50 }, "collector.max_redirects"},
		{"zero item cap", func(c *Config) { c.Collector.MaxItemsPerFeed = 0 }, "collector.max_items_per_feed"},
		{
			"private networks in production",
			func(c *Config) { c.App.Environment = "production"; c.Collector.AllowPrivateNetworks = true },
			"collector.allow_private_networks",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Default()
			tc.mutate(cfg)

			err := cfg.Validate()
			if err == nil {
				t.Fatalf("expected a validation error mentioning %s", tc.wantMsg)
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Errorf("error should mention %s, got: %v", tc.wantMsg, err)
			}
		})
	}
}

func TestValidateReportsAllProblemsAtOnce(t *testing.T) {
	cfg := Default()
	cfg.Server.Port = 0
	cfg.Logging.Level = "verbose"

	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected a validation error")
	}
	if !strings.Contains(err.Error(), "server.port") || !strings.Contains(err.Error(), "logging.level") {
		t.Errorf("both problems should be reported, got: %v", err)
	}
}

func TestRedactedURIHidesCredentials(t *testing.T) {
	m := Mongo{URI: "mongodb://admin:sup3rs3cret@localhost:27017/news"}

	got := m.RedactedURI()
	if strings.Contains(got, "sup3rs3cret") {
		t.Fatalf("RedactedURI leaked the password: %q", got)
	}
	if !strings.Contains(got, "localhost:27017") {
		t.Errorf("RedactedURI should keep the host, got %q", got)
	}
}

func TestServerAddress(t *testing.T) {
	s := Server{Host: "127.0.0.1", Port: 8080}
	if got := s.Address(); got != "127.0.0.1:8080" {
		t.Errorf("Address() = %q, want %q", got, "127.0.0.1:8080")
	}
}

func TestCollectorEnvOverrides(t *testing.T) {
	t.Setenv(EnvPrefix+"COLLECTOR_MAX_RESPONSE_BYTES", "2097152")
	t.Setenv(EnvPrefix+"COLLECTOR_MAX_REDIRECTS", "2")
	t.Setenv(EnvPrefix+"COLLECTOR_ALLOW_PRIVATE_NETWORKS", "true")
	t.Setenv(EnvPrefix+"COLLECTOR_REQUEST_TIMEOUT", "7s")

	cfg, err := Load("")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Collector.MaxResponseBytes != 2<<20 {
		t.Errorf("MaxResponseBytes = %d, want 2097152", cfg.Collector.MaxResponseBytes)
	}
	if cfg.Collector.MaxRedirects != 2 {
		t.Errorf("MaxRedirects = %d, want 2", cfg.Collector.MaxRedirects)
	}
	if !cfg.Collector.AllowPrivateNetworks {
		t.Error("AllowPrivateNetworks = false, want the env override applied")
	}
	if cfg.Collector.RequestTimeout != 7*time.Second {
		t.Errorf("RequestTimeout = %s, want 7s", cfg.Collector.RequestTimeout)
	}
}

func TestCollectorMalformedBooleanIsReported(t *testing.T) {
	t.Setenv(EnvPrefix+"COLLECTOR_ALLOW_PRIVATE_NETWORKS", "yes-please")

	_, err := Load("")
	if err == nil || !strings.Contains(err.Error(), "COLLECTOR_ALLOW_PRIVATE_NETWORKS") {
		t.Fatalf("error should name the malformed variable, got: %v", err)
	}
}

func TestSchedulerEnvOverrides(t *testing.T) {
	t.Setenv(EnvPrefix+"SCHEDULER_ENABLED", "false")
	t.Setenv(EnvPrefix+"SCHEDULER_INTERVAL", "30s")
	t.Setenv(EnvPrefix+"SCHEDULER_BATCH_SIZE", "10")
	t.Setenv(EnvPrefix+"SCHEDULER_MAX_CONCURRENT", "2")
	t.Setenv(EnvPrefix+"SCHEDULER_LOCK_TTL", "90s")

	cfg, err := Load("")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Scheduler.Enabled {
		t.Error("Enabled = true, want the env override applied")
	}
	if cfg.Scheduler.Interval != 30*time.Second || cfg.Scheduler.LockTTL != 90*time.Second {
		t.Errorf("durations = %s / %s, want 30s / 90s", cfg.Scheduler.Interval, cfg.Scheduler.LockTTL)
	}
	if cfg.Scheduler.BatchSize != 10 || cfg.Scheduler.MaxConcurrent != 2 {
		t.Errorf("batch/concurrency = %d/%d, want 10/2", cfg.Scheduler.BatchSize, cfg.Scheduler.MaxConcurrent)
	}
}

func TestSchedulerValidation(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{
			name:    "no interval",
			mutate:  func(c *Config) { c.Scheduler.Interval = 0 },
			wantErr: "scheduler.interval",
		},
		{
			name:    "batch beyond the cap",
			mutate:  func(c *Config) { c.Scheduler.BatchSize = maxSchedulerBatchSize + 1 },
			wantErr: "scheduler.batch_size",
		},
		{
			name:    "concurrency beyond the cap",
			mutate:  func(c *Config) { c.Scheduler.MaxConcurrent = maxSchedulerConcurrency + 1 },
			wantErr: "scheduler.max_concurrent",
		},
		{
			// A lease that expires while its own fetch is still running is the
			// exact collision the lease exists to prevent.
			name:    "lease shorter than a fetch",
			mutate:  func(c *Config) { c.Scheduler.LockTTL = c.Collector.RequestTimeout },
			wantErr: "scheduler.lock_ttl",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := Default()
			tt.mutate(cfg)

			err := cfg.Validate()
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %v, want it to name %s", err, tt.wantErr)
			}
		})
	}
}

// A disabled scheduler is not configured at all, so its settings must not be
// able to stop the process starting.
func TestDisabledSchedulerIsNotValidated(t *testing.T) {
	cfg := Default()
	cfg.Scheduler.Enabled = false
	cfg.Scheduler.Interval = 0
	cfg.Scheduler.BatchSize = 0
	cfg.Scheduler.MaxConcurrent = 0
	cfg.Scheduler.LockTTL = 0

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}
