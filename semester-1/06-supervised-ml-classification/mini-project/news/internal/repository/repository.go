// Package repository declares the persistence contracts the application depends
// on, together with the storage-neutral errors those contracts may return.
// Implementations live in subpackages; nothing here imports a database driver,
// so services and CLIs can be tested against an in-memory fake.
package repository

import (
	"context"
	"errors"

	"github.com/riaz/newscollector/internal/domain"
)

// Storage-neutral sentinels. Implementations must translate driver-specific
// failures into these so no caller has to recognise a MongoDB error code.
var (
	// ErrNotFound means no record matched the identifier or filter.
	ErrNotFound = errors.New("repository: not found")

	// ErrDuplicate means a unique constraint rejected the write. It is the
	// authority on duplicates: a read-then-write check would race.
	ErrDuplicate = errors.New("repository: duplicate key")
)

// SourceRepository persists configured feeds.
type SourceRepository interface {
	// Create stores a new source, returning ErrDuplicate if its feed URL is taken.
	Create(ctx context.Context, s *domain.Source) error

	// GetByID returns the source with the given id, or ErrNotFound.
	GetByID(ctx context.Context, id string) (*domain.Source, error)

	// GetByFeedURL returns the source registered for a feed URL, or ErrNotFound.
	GetByFeedURL(ctx context.Context, feedURL string) (*domain.Source, error)

	// List returns the page of sources matching filter, plus the total match count.
	List(ctx context.Context, filter domain.SourceFilter) (domain.SourcePage, error)

	// Update replaces the stored source, returning ErrNotFound if the id is
	// unknown or ErrDuplicate if the new feed URL collides with another source.
	Update(ctx context.Context, s *domain.Source) error

	// Delete removes a source, returning ErrNotFound if the id is unknown.
	Delete(ctx context.Context, id string) error
}
