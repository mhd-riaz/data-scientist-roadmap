// Package config loads application configuration from three layers, in
// increasing order of precedence: built-in defaults, an optional YAML file, and
// environment variables prefixed with EnvPrefix.
package config

import (
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// EnvPrefix is prepended to every environment variable that overrides a config value.
const EnvPrefix = "NEWS_"

// DefaultPath is the config file location used when none is supplied.
const DefaultPath = "configs/config.yaml"

// Config is the fully resolved application configuration.
type Config struct {
	App     App     `yaml:"app"`
	Server  Server  `yaml:"server"`
	Mongo   Mongo   `yaml:"mongo"`
	Logging Logging `yaml:"logging"`
}

// App holds process identity settings.
type App struct {
	Name        string `yaml:"name"`
	Environment string `yaml:"environment"`
}

// Server holds HTTP listener settings.
type Server struct {
	Host              string        `yaml:"host"`
	Port              int           `yaml:"port"`
	ReadHeaderTimeout time.Duration `yaml:"read_header_timeout"`
	ReadTimeout       time.Duration `yaml:"read_timeout"`
	WriteTimeout      time.Duration `yaml:"write_timeout"`
	IdleTimeout       time.Duration `yaml:"idle_timeout"`
	ShutdownTimeout   time.Duration `yaml:"shutdown_timeout"`
	MaxHeaderBytes    int           `yaml:"max_header_bytes"`
}

// Mongo holds MongoDB connection settings.
type Mongo struct {
	URI                    string        `yaml:"uri"`
	Database               string        `yaml:"database"`
	ConnectTimeout         time.Duration `yaml:"connect_timeout"`
	ServerSelectionTimeout time.Duration `yaml:"server_selection_timeout"`
	OperationTimeout       time.Duration `yaml:"operation_timeout"`
	MaxPoolSize            uint64        `yaml:"max_pool_size"`
	MinPoolSize            uint64        `yaml:"min_pool_size"`
}

// Logging holds structured-logging settings.
type Logging struct {
	Level  string `yaml:"level"`
	Format string `yaml:"format"`
}

// Address returns the host:port the HTTP server should listen on.
func (s Server) Address() string {
	return net.JoinHostPort(s.Host, strconv.Itoa(s.Port))
}

// RedactedURI returns the MongoDB URI with any embedded credentials replaced,
// so the connection target can be logged without leaking a password.
func (m Mongo) RedactedURI() string {
	u, err := url.Parse(m.URI)
	if err != nil {
		return "<unparsable mongo uri>"
	}
	if u.User != nil {
		u.User = url.User("redacted")
	}
	return u.String()
}

// Default returns the built-in configuration used as the base of every load.
func Default() *Config {
	return &Config{
		App: App{
			Name:        "news-collector",
			Environment: "development",
		},
		Server: Server{
			Host:              "0.0.0.0",
			Port:              8080,
			ReadHeaderTimeout: 5 * time.Second,
			ReadTimeout:       15 * time.Second,
			WriteTimeout:      30 * time.Second,
			IdleTimeout:       60 * time.Second,
			ShutdownTimeout:   15 * time.Second,
			MaxHeaderBytes:    1 << 20,
		},
		Mongo: Mongo{
			URI:                    "mongodb://localhost:27017",
			Database:               "news",
			ConnectTimeout:         10 * time.Second,
			ServerSelectionTimeout: 10 * time.Second,
			OperationTimeout:       30 * time.Second,
			MaxPoolSize:            50,
			MinPoolSize:            0,
		},
		Logging: Logging{
			Level:  "info",
			Format: "json",
		},
	}
}

// Load resolves configuration from defaults, the YAML file at path when it is
// non-empty, and environment variables, then validates the result. A path that
// does not exist is reported as an error; pass "" to skip the file layer.
func Load(path string) (*Config, error) {
	cfg := Default()

	if path != "" {
		if err := cfg.loadFile(path); err != nil {
			return nil, err
		}
	}
	if err := cfg.applyEnv(); err != nil {
		return nil, err
	}
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}
	return cfg, nil
}

func (c *Config) loadFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open config file: %w", err)
	}
	defer func() { _ = f.Close() }()

	dec := yaml.NewDecoder(f)
	dec.KnownFields(true) // reject typo'd keys instead of silently ignoring them
	if err := dec.Decode(c); err != nil && !errors.Is(err, io.EOF) {
		return fmt.Errorf("parse config file %q: %w", path, err)
	}
	return nil
}

func (c *Config) applyEnv() error {
	b := &envBinder{}

	b.str("APP_NAME", &c.App.Name)
	b.str("APP_ENVIRONMENT", &c.App.Environment)

	b.str("SERVER_HOST", &c.Server.Host)
	b.integer("SERVER_PORT", &c.Server.Port)
	b.duration("SERVER_READ_HEADER_TIMEOUT", &c.Server.ReadHeaderTimeout)
	b.duration("SERVER_READ_TIMEOUT", &c.Server.ReadTimeout)
	b.duration("SERVER_WRITE_TIMEOUT", &c.Server.WriteTimeout)
	b.duration("SERVER_IDLE_TIMEOUT", &c.Server.IdleTimeout)
	b.duration("SERVER_SHUTDOWN_TIMEOUT", &c.Server.ShutdownTimeout)
	b.integer("SERVER_MAX_HEADER_BYTES", &c.Server.MaxHeaderBytes)

	b.str("MONGO_URI", &c.Mongo.URI)
	b.str("MONGO_DATABASE", &c.Mongo.Database)
	b.duration("MONGO_CONNECT_TIMEOUT", &c.Mongo.ConnectTimeout)
	b.duration("MONGO_SERVER_SELECTION_TIMEOUT", &c.Mongo.ServerSelectionTimeout)
	b.duration("MONGO_OPERATION_TIMEOUT", &c.Mongo.OperationTimeout)
	b.unsigned("MONGO_MAX_POOL_SIZE", &c.Mongo.MaxPoolSize)
	b.unsigned("MONGO_MIN_POOL_SIZE", &c.Mongo.MinPoolSize)

	b.str("LOGGING_LEVEL", &c.Logging.Level)
	b.str("LOGGING_FORMAT", &c.Logging.Format)

	return errors.Join(b.errs...)
}

// Validate reports every configuration problem at once rather than only the first.
func (c *Config) Validate() error {
	var errs []error

	if strings.TrimSpace(c.App.Name) == "" {
		errs = append(errs, errors.New("app.name must not be empty"))
	}
	errs = append(errs, oneOf("app.environment", c.App.Environment, "development", "staging", "production"))

	if c.Server.Port < 1 || c.Server.Port > 65535 {
		errs = append(errs, fmt.Errorf("server.port must be between 1 and 65535, got %d", c.Server.Port))
	}
	if strings.TrimSpace(c.Server.Host) == "" {
		errs = append(errs, errors.New("server.host must not be empty"))
	}
	if c.Server.MaxHeaderBytes <= 0 {
		errs = append(errs, errors.New("server.max_header_bytes must be greater than zero"))
	}
	errs = append(errs,
		positive("server.read_header_timeout", c.Server.ReadHeaderTimeout),
		positive("server.read_timeout", c.Server.ReadTimeout),
		positive("server.write_timeout", c.Server.WriteTimeout),
		positive("server.idle_timeout", c.Server.IdleTimeout),
		positive("server.shutdown_timeout", c.Server.ShutdownTimeout),
		positive("mongo.connect_timeout", c.Mongo.ConnectTimeout),
		positive("mongo.server_selection_timeout", c.Mongo.ServerSelectionTimeout),
		positive("mongo.operation_timeout", c.Mongo.OperationTimeout),
		validateMongoURI(c.Mongo.URI),
		validateDatabaseName(c.Mongo.Database),
	)

	if c.Mongo.MaxPoolSize == 0 {
		errs = append(errs, errors.New("mongo.max_pool_size must be greater than zero"))
	} else if c.Mongo.MinPoolSize > c.Mongo.MaxPoolSize {
		errs = append(errs, fmt.Errorf("mongo.min_pool_size (%d) must not exceed mongo.max_pool_size (%d)",
			c.Mongo.MinPoolSize, c.Mongo.MaxPoolSize))
	}

	errs = append(errs,
		oneOf("logging.level", c.Logging.Level, "debug", "info", "warn", "error"),
		oneOf("logging.format", c.Logging.Format, "json", "text"),
	)

	return errors.Join(errs...)
}

func validateMongoURI(uri string) error {
	if strings.TrimSpace(uri) == "" {
		return errors.New("mongo.uri must not be empty")
	}
	if !strings.HasPrefix(uri, "mongodb://") && !strings.HasPrefix(uri, "mongodb+srv://") {
		return errors.New(`mongo.uri must start with "mongodb://" or "mongodb+srv://"`)
	}
	if _, err := url.Parse(uri); err != nil {
		return fmt.Errorf("mongo.uri is not a valid URI: %w", err)
	}
	return nil
}

// invalidDatabaseChars are the characters MongoDB forbids in a database name.
const invalidDatabaseChars = "/\\. \"$*<>:|?"

func validateDatabaseName(name string) error {
	if strings.TrimSpace(name) == "" {
		return errors.New("mongo.database must not be empty")
	}
	if strings.ContainsAny(name, invalidDatabaseChars) || strings.ContainsRune(name, 0) {
		return fmt.Errorf("mongo.database %q contains characters that MongoDB forbids (%s)", name, invalidDatabaseChars)
	}
	if len(name) > 63 {
		return fmt.Errorf("mongo.database must be at most 63 bytes, got %d", len(name))
	}
	return nil
}

func positive(field string, d time.Duration) error {
	if d <= 0 {
		return fmt.Errorf("%s must be greater than zero, got %s", field, d)
	}
	return nil
}

func oneOf(field, value string, allowed ...string) error {
	for _, a := range allowed {
		if value == a {
			return nil
		}
	}
	return fmt.Errorf("%s must be one of %s, got %q", field, strings.Join(allowed, ", "), value)
}

// envBinder applies environment overrides, collecting parse errors so a single
// load reports every malformed variable instead of only the first.
type envBinder struct {
	errs []error
}

func (b *envBinder) lookup(key string) (string, bool) {
	raw, ok := os.LookupEnv(EnvPrefix + key)
	if !ok {
		return "", false
	}
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return "", false
	}
	return trimmed, true
}

func (b *envBinder) fail(key string, err error) {
	b.errs = append(b.errs, fmt.Errorf("%s%s: %w", EnvPrefix, key, err))
}

func (b *envBinder) str(key string, dst *string) {
	if v, ok := b.lookup(key); ok {
		*dst = v
	}
}

func (b *envBinder) integer(key string, dst *int) {
	v, ok := b.lookup(key)
	if !ok {
		return
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		b.fail(key, fmt.Errorf("expected an integer, got %q", v))
		return
	}
	*dst = n
}

func (b *envBinder) unsigned(key string, dst *uint64) {
	v, ok := b.lookup(key)
	if !ok {
		return
	}
	n, err := strconv.ParseUint(v, 10, 64)
	if err != nil {
		b.fail(key, fmt.Errorf("expected a non-negative integer, got %q", v))
		return
	}
	*dst = n
}

func (b *envBinder) duration(key string, dst *time.Duration) {
	v, ok := b.lookup(key)
	if !ok {
		return
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		b.fail(key, fmt.Errorf("expected a duration such as 15s, got %q", v))
		return
	}
	*dst = d
}
