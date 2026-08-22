// Command api serves the HTTP interface of the news collection system.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/riaz/newscollector/internal/app"
	"github.com/riaz/newscollector/internal/config"
	"github.com/riaz/newscollector/internal/handler"
	"github.com/riaz/newscollector/internal/mongodb"
	"github.com/riaz/newscollector/internal/observability"
	mongorepo "github.com/riaz/newscollector/internal/repository/mongo"
	"github.com/riaz/newscollector/internal/scheduler"
	"github.com/riaz/newscollector/internal/service"
)

// version is overridden at build time with -ldflags "-X main.version=...".
var version = "dev"

// readinessCheckTimeout bounds the dependency probe inside /health/ready.
const readinessCheckTimeout = 2 * time.Second

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	configPath := flag.String("config", defaultConfigPath(), "path to the YAML configuration file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}

	logger, err := observability.NewLogger(cfg.Logging.Level, cfg.Logging.Format, os.Stdout)
	if err != nil {
		return err
	}
	logger = logger.With(
		"service", cfg.App.Name,
		"environment", cfg.App.Environment,
		"version", version,
	)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	mongoClient, err := mongodb.Connect(mongodb.Settings{
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

	// A database outage must not stop the process from starting: liveness stays
	// green and readiness reports 503 until MongoDB comes back.
	pingCtx, cancelPing := context.WithTimeout(ctx, cfg.Mongo.ServerSelectionTimeout)
	if err := mongoClient.Ping(pingCtx); err != nil {
		logger.Warn("mongodb unreachable at startup; readiness will report not ready",
			"uri", cfg.Mongo.RedactedURI(), "error", err)
	} else {
		logger.Info("connected to mongodb", "uri", cfg.Mongo.RedactedURI(), "database", cfg.Mongo.Database)
	}
	cancelPing()

	sourceService := service.NewSourceService(mongorepo.NewSourceRepository(mongoClient.Database()), time.Now)
	articleService := service.NewArticleService(mongorepo.NewArticleRepository(mongoClient.Database()))

	collectionService, err := app.NewCollectionService(cfg, mongoClient.Database(), time.Now, logger)
	if err != nil {
		return err
	}

	srv := &http.Server{
		Addr: cfg.Server.Address(),
		Handler: handler.NewRouter(
			handler.NewHealth(mongoClient, readinessCheckTimeout, version, logger),
			handler.NewSource(sourceService, logger),
			handler.NewCollectionRun(collectionService, logger),
			handler.NewArticle(articleService, logger),
			logger,
		),
		ReadHeaderTimeout: cfg.Server.ReadHeaderTimeout,
		ReadTimeout:       cfg.Server.ReadTimeout,
		WriteTimeout:      cfg.Server.WriteTimeout,
		IdleTimeout:       cfg.Server.IdleTimeout,
		MaxHeaderBytes:    cfg.Server.MaxHeaderBytes,
	}

	// The scheduler shares the process with the API. Per-source leases are what
	// make that safe with more than one replica running.
	var schedulerDone sync.WaitGroup
	if cfg.Scheduler.Enabled {
		sch := scheduler.New(collectionService, scheduler.Config{
			Interval:      cfg.Scheduler.Interval,
			BatchSize:     cfg.Scheduler.BatchSize,
			MaxConcurrent: cfg.Scheduler.MaxConcurrent,
		}, logger)

		schedulerDone.Add(1)
		go func() {
			defer schedulerDone.Done()
			if err := sch.Run(ctx); err != nil {
				logger.Error("scheduler stopped early", "error", err)
			}
		}()
	} else {
		logger.Warn("scheduler is disabled; feeds will only be collected by the collector command")
	}

	serverErr := make(chan error, 1)
	go func() {
		logger.Info("http server listening", "address", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
			return
		}
		serverErr <- nil
	}()

	var runErr error
	select {
	case err := <-serverErr:
		runErr = err
	case <-ctx.Done():
		logger.Info("shutdown signal received", "timeout", cfg.Server.ShutdownTimeout)
	}

	// Detach from the cancelled signal context so in-flight requests get their
	// full grace period.
	shutdownCtx, cancelShutdown := context.WithTimeout(context.WithoutCancel(ctx), cfg.Server.ShutdownTimeout)
	defer cancelShutdown()

	shutdownErr := srv.Shutdown(shutdownCtx)
	if shutdownErr != nil {
		logger.Error("graceful shutdown failed", "error", shutdownErr)
	}

	// The scheduler stops with the signal context; waiting for it here means a
	// collection in flight finishes and gives its lease back before the MongoDB
	// connection it needs is closed.
	stop()
	schedulerDone.Wait()

	if err := mongoClient.Close(shutdownCtx); err != nil {
		logger.Error("closing mongodb failed", "error", err)
	}

	logger.Info("shutdown complete")
	return errors.Join(runErr, shutdownErr)
}

// defaultConfigPath lets the container point at a mounted config without flags.
func defaultConfigPath() string {
	if p := os.Getenv(config.EnvPrefix + "CONFIG_PATH"); p != "" {
		return p
	}
	return config.DefaultPath
}
