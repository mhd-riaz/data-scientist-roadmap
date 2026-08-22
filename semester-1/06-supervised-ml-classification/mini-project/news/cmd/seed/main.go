// Command seed applies a declarative list of sources to the database. It is
// idempotent: running it twice creates nothing the second time, so it is safe to
// re-run after editing the file.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/riaz/newscollector/internal/config"
	"github.com/riaz/newscollector/internal/domain"
	"github.com/riaz/newscollector/internal/mongodb"
	"github.com/riaz/newscollector/internal/observability"
	mongorepo "github.com/riaz/newscollector/internal/repository/mongo"
	"github.com/riaz/newscollector/internal/service"
)

// version is overridden at build time with -ldflags "-X main.version=...".
var version = "dev"

// defaultSourcesPath is the seed file used when none is given.
const defaultSourcesPath = "configs/sources.yaml"

// maxSeedFileBytes caps the seed file. It is operator-supplied, but a runaway
// file should fail fast rather than be read into memory in full.
const maxSeedFileBytes = 4 << 20

// seedFile is the on-disk seed format.
type seedFile struct {
	Sources []seedSource `yaml:"sources"`
}

// seedSource mirrors domain.SourceInput. Optional fields are pointers so an
// omitted key keeps the stored value instead of resetting it to a default.
type seedSource struct {
	Name                 string `yaml:"name"`
	FeedURL              string `yaml:"feed_url"`
	Type                 string `yaml:"type"`
	Language             string `yaml:"language"`
	Country              string `yaml:"country"`
	State                string `yaml:"state"`
	City                 string `yaml:"city"`
	Enabled              *bool  `yaml:"enabled"`
	Priority             *int   `yaml:"priority"`
	FetchIntervalSeconds *int   `yaml:"fetch_interval_seconds"`
}

func (s seedSource) toInput() domain.SourceInput {
	return domain.SourceInput{
		Name:                 s.Name,
		FeedURL:              s.FeedURL,
		Type:                 domain.SourceType(s.Type),
		Language:             s.Language,
		Country:              s.Country,
		State:                s.State,
		City:                 s.City,
		Enabled:              s.Enabled,
		Priority:             s.Priority,
		FetchIntervalSeconds: s.FetchIntervalSeconds,
	}
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	configPath := flag.String("config", defaultConfigPath(), "path to the YAML configuration file")
	envPath := flag.String("env", config.EnvFilePath(), "path to the dotenv file holding secrets")
	sourcesPath := flag.String("sources", defaultSourcesPath, "path to the YAML seed file")
	dryRun := flag.Bool("dry-run", false, "validate the seed file without writing to the database")
	flag.Parse()

	if err := config.LoadEnvFile(*envPath); err != nil {
		return err
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}

	logger, err := observability.NewLogger(cfg.Logging.Level, cfg.Logging.Format, os.Stdout)
	if err != nil {
		return err
	}
	logger = logger.With("service", cfg.App.Name, "command", "seed", "version", version)

	seed, err := readSeedFile(*sourcesPath)
	if err != nil {
		return err
	}
	if len(seed.Sources) == 0 {
		return fmt.Errorf("seed file %q declares no sources", *sourcesPath)
	}

	// Validating everything before the first write stops a typo halfway down the
	// file from leaving the database half-seeded.
	inputs, err := validateAll(seed.Sources)
	if err != nil {
		return err
	}
	if *dryRun {
		logger.Info("seed file is valid", "path", *sourcesPath, "sources", len(inputs))
		return nil
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	client, err := mongodb.Connect(mongodb.Settings{
		URI:                    cfg.Mongo.URI,
		Database:               cfg.Mongo.Database,
		AppName:                cfg.App.Name,
		ConnectTimeout:         cfg.Mongo.ConnectTimeout,
		ServerSelectionTimeout: cfg.Mongo.ServerSelectionTimeout,
		OperationTimeout:       cfg.Mongo.OperationTimeout,
		MaxPoolSize:            cfg.Mongo.MaxPoolSize,
		MinPoolSize:            cfg.Mongo.MinPoolSize,
	})
	if err != nil {
		return err
	}
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		defer cancel()
		if err := client.Close(closeCtx); err != nil {
			logger.Error("closing mongodb failed", "error", err)
		}
	}()

	if err := client.Ping(ctx); err != nil {
		return fmt.Errorf("mongodb is not reachable at %s: %w", cfg.Mongo.RedactedURI(), err)
	}

	svc := service.NewSourceService(mongorepo.NewSourceRepository(client.Database()), time.Now)

	var created, updated int
	for _, in := range inputs {
		src, wasCreated, err := svc.Ensure(ctx, in)
		if err != nil {
			return fmt.Errorf("seed %q: %w", in.FeedURL, err)
		}
		if wasCreated {
			created++
		} else {
			updated++
		}
		logger.Info("seeded source", "id", src.ID, "feed_url", src.FeedURL, "created", wasCreated)
	}

	logger.Info("seeding complete", "created", created, "updated", updated, "total", len(inputs))
	return nil
}

// readSeedFile parses the seed file, rejecting unknown keys so a typo fails fast
// rather than being silently ignored.
func readSeedFile(path string) (*seedFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open seed file: %w", err)
	}
	defer func() { _ = f.Close() }()

	return parseSeedFile(io.LimitReader(f, maxSeedFileBytes))
}

func parseSeedFile(r io.Reader) (*seedFile, error) {
	var seed seedFile
	dec := yaml.NewDecoder(r)
	dec.KnownFields(true)
	if err := dec.Decode(&seed); err != nil && !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("parse seed file: %w", err)
	}
	return &seed, nil
}

// validateAll converts and validates every entry, reporting the file position of
// each failure so an operator can find it.
func validateAll(sources []seedSource) ([]domain.SourceInput, error) {
	inputs := make([]domain.SourceInput, 0, len(sources))
	seen := make(map[string]int, len(sources))
	var errs []error

	for i, s := range sources {
		in := s.toInput()
		if _, err := domain.NewSource(in, time.Now()); err != nil {
			errs = append(errs, fmt.Errorf("sources[%d] (%s): %w", i, s.Name, err))
			continue
		}
		if first, duplicate := seen[s.FeedURL]; duplicate {
			errs = append(errs, fmt.Errorf("sources[%d]: feed_url %q already declared at sources[%d]", i, s.FeedURL, first))
			continue
		}
		seen[s.FeedURL] = i
		inputs = append(inputs, in)
	}

	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}
	return inputs, nil
}

// defaultConfigPath lets the container point at a mounted config without flags.
func defaultConfigPath() string {
	if p := os.Getenv(config.EnvPrefix + "CONFIG_PATH"); p != "" {
		return p
	}
	return config.DefaultPath
}
