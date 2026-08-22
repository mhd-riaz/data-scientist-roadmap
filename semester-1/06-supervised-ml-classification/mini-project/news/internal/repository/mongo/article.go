package mongo

import (
	"context"
	"errors"
	"fmt"

	"go.mongodb.org/mongo-driver/v2/bson"
	driver "go.mongodb.org/mongo-driver/v2/mongo"

	"github.com/riaz/newscollector/internal/domain"
	"github.com/riaz/newscollector/internal/mongodb"
	"github.com/riaz/newscollector/internal/repository"
)

// ArticleRepository stores collected articles in the articles collection.
type ArticleRepository struct {
	coll *driver.Collection
}

// NewArticleRepository binds the repository to the articles collection.
func NewArticleRepository(db *driver.Database) *ArticleRepository {
	return &ArticleRepository{coll: db.Collection(mongodb.CollectionArticles)}
}

var _ repository.ArticleRepository = (*ArticleRepository)(nil)

// Create inserts an article. Any of uq_dedup_id, uq_normalized_url and
// uq_source_feed_guid may reject it; all three mean the same thing to a caller,
// so all three come back as ErrDuplicate.
func (r *ArticleRepository) Create(ctx context.Context, a *domain.Article) error {
	if _, err := r.coll.InsertOne(ctx, a); err != nil {
		if driver.IsDuplicateKeyError(err) {
			return repository.ErrDuplicate
		}
		return fmt.Errorf("mongo: insert article: %w", err)
	}
	return nil
}

// FindByIdentity looks the article up by each of its keys in turn and returns
// the first match.
//
// The keys are queried one at a time rather than combined into a single $or so
// the documented precedence is the precedence that actually runs, and so every
// query is served by exactly one index. The common case — a feed re-publishing
// an article this collector already has — matches on the first query.
func (r *ArticleRepository) FindByIdentity(ctx context.Context, identity domain.ArticleIdentity) (*domain.Article, error) {
	filters := make([]bson.D, 0, 4)

	if identity.NormalizedURL != "" {
		filters = append(filters, bson.D{{Key: "normalized_url", Value: identity.NormalizedURL}})
	}
	if identity.CanonicalURL != "" {
		filters = append(filters, bson.D{{Key: "canonical_url", Value: identity.CanonicalURL}})
	}
	if identity.SourceID != "" && identity.FeedGUID != "" {
		filters = append(filters, bson.D{
			{Key: "source_id", Value: identity.SourceID},
			{Key: "feed_guid", Value: identity.FeedGUID},
		})
	}
	// The content hash is only conclusive within one source: two sources may
	// legitimately syndicate the same story, and both are worth keeping.
	if identity.SourceID != "" && identity.ContentHash != "" {
		filters = append(filters, bson.D{
			{Key: "source_id", Value: identity.SourceID},
			{Key: "content_hash", Value: identity.ContentHash},
		})
	}

	for _, filter := range filters {
		a, err := r.findOne(ctx, filter)
		if err == nil {
			return a, nil
		}
		if !errors.Is(err, repository.ErrNotFound) {
			return nil, err
		}
	}
	return nil, repository.ErrNotFound
}

func (r *ArticleRepository) findOne(ctx context.Context, filter bson.D) (*domain.Article, error) {
	var a domain.Article
	if err := r.coll.FindOne(ctx, filter).Decode(&a); err != nil {
		if errors.Is(err, driver.ErrNoDocuments) {
			return nil, repository.ErrNotFound
		}
		return nil, fmt.Errorf("mongo: find article: %w", err)
	}
	return &a, nil
}
