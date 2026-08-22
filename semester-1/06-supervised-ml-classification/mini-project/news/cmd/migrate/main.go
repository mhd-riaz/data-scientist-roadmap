// Command migrate creates the application's MongoDB collections and indexes.
// It is idempotent and safe to run before every deployment.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/riaz/newscollector/internal/config"
	"github.com/riaz/newscollector/internal/mongodb"
	"github.com/riaz/newscollector/internal/observability"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	configPath := flag.String("config", defaultConfigPath(), "path to the YAML configuration file")
	timeout := flag.Duration("timeout", 2*time.Minute, "overall time budget for the migration")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}

	logger, err := observability.NewLogger(cfg.Logging.Level, cfg.Logging.Format, os.Stdout)
	if err != nil {
		return err
	}
	logger = logger.With("command", "migrate", "database", cfg.Mongo.Database)

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	client, err := mongodb.Connect(mongodb.Settings{
		URI:                    cfg.Mongo.URI,
		Database:               cfg.Mongo.Database,
		AppName:                cfg.App.Name + "-migrate",
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
		closeCtx, closeCancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		defer closeCancel()
		if err := client.Close(closeCtx); err != nil {
			logger.Error("closing mongodb failed", "error", err)
		}
	}()

	// Unlike the API, a migration has nothing useful to do without the database.
	if err := client.Ping(ctx); err != nil {
		return fmt.Errorf("cannot reach mongodb at %s: %w", cfg.Mongo.RedactedURI(), err)
	}

	created, err := mongodb.EnsureCollections(ctx, client.Database())
	if err != nil {
		return err
	}
	logger.Info("collections ensured", "created", created, "total", len(mongodb.Collections()))

	// Superseded indexes go before the new ones: MongoDB refuses to recreate an
	// existing name with different keys.
	dropped, err := mongodb.DropObsoleteIndexes(ctx, client.Database())
	if err != nil {
		return err
	}
	for collection, names := range dropped {
		logger.Info("obsolete indexes dropped", "collection", collection, "indexes", names)
	}

	applied, err := mongodb.EnsureIndexes(ctx, client.Database())
	if err != nil {
		return err
	}
	for _, ci := range mongodb.IndexPlan() {
		logger.Info("indexes ensured", "collection", ci.Collection, "indexes", applied[ci.Collection])
	}

	logger.Info("migration complete")
	return nil
}

func defaultConfigPath() string {
	if p := os.Getenv(config.EnvPrefix + "CONFIG_PATH"); p != "" {
		return p
	}
	return config.DefaultPath
}
