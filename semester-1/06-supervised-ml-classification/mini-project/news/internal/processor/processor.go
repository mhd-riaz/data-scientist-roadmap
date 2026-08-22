// Package processor turns the items a collector read out of a feed into stored
// articles. It is the only place that decides whether something has been seen
// before, so the deduplication order lives here and in the index plan that
// enforces it, nowhere else.
package processor

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/riaz/newscollector/internal/domain"
	"github.com/riaz/newscollector/internal/repository"
)

// Clock supplies the current time. Injecting it keeps behaviour deterministic
// under test instead of depending on the wall clock.
type Clock func() time.Time

// Result counts what one batch did. The counts always add up to Items, so a
// feed that stores nothing still says why: all duplicates, or all unusable.
type Result struct {
	Items      int
	Stored     int
	Duplicates int
	Invalid    int
}

// Processor stores collected items.
type Processor struct {
	articles repository.ArticleRepository
	clock    Clock
}

// New wires the processor. A nil clock defaults to time.Now.
func New(articles repository.ArticleRepository, clock Clock) *Processor {
	if clock == nil {
		clock = time.Now
	}
	return &Processor{articles: articles, clock: clock}
}

// Process normalises, deduplicates and stores a batch of items collected from
// src, in that order, one item at a time.
//
// Items are handled sequentially so that two entries duplicating each other
// within the same batch are caught by the same lookup that catches a duplicate
// of an earlier collection. An item the model rejects is counted and skipped,
// because one malformed entry must not cost the rest of the feed. A repository
// failure, on the other hand, stops the batch: it means the database is not
// answering, and the counts so far are returned with the error so the caller
// can record what did get through.
func (p *Processor) Process(ctx context.Context, src *domain.Source, items []domain.FeedItem) (Result, error) {
	result := Result{Items: len(items)}
	if src == nil {
		return result, errors.New("processor: nil source")
	}

	for _, item := range items {
		article, err := domain.NewArticle(*src, item, p.clock())
		if err != nil {
			if errors.Is(err, domain.ErrValidation) {
				result.Invalid++
				continue
			}
			return result, err
		}

		stored, err := p.store(ctx, article)
		switch {
		case err != nil:
			return result, err
		case stored:
			result.Stored++
		default:
			result.Duplicates++
		}
	}

	return result, nil
}

// store reports whether the article was new. The lookup answers the common case
// without a write, and the unique indexes settle the race the lookup cannot:
// between the read and the insert, another collector may have stored the same
// article, and a duplicate key there is the expected outcome rather than a fault.
func (p *Processor) store(ctx context.Context, article *domain.Article) (bool, error) {
	_, err := p.articles.FindByIdentity(ctx, article.Identity())
	switch {
	case err == nil:
		return false, nil
	case !errors.Is(err, repository.ErrNotFound):
		return false, fmt.Errorf("processor: look up article: %w", err)
	}

	switch err := p.articles.Create(ctx, article); {
	case err == nil:
		return true, nil
	case errors.Is(err, repository.ErrDuplicate):
		return false, nil
	default:
		return false, fmt.Errorf("processor: store article: %w", err)
	}
}
