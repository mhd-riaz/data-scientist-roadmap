package mongodb

import (
	"context"
	"errors"
	"fmt"

	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

// namespaceExistsCode is the MongoDB error returned when a collection already exists.
const namespaceExistsCode = 48

// CollectionIndexes binds a collection to the indexes it requires.
type CollectionIndexes struct {
	Collection string
	Models     []mongo.IndexModel
}

// IndexPlan is the full set of indexes the application depends on.
//
// The unique indexes on articles are the database-level enforcement of the
// deduplication order: normalized URL, then canonical URL, then source plus feed
// GUID. content_hash is the final fallback lookup and is deliberately not unique,
// because two sources may legitimately syndicate byte-identical content.
func IndexPlan() []CollectionIndexes {
	return []CollectionIndexes{
		{
			Collection: CollectionSources,
			Models: []mongo.IndexModel{
				{
					Keys:    bson.D{{Key: "feed_url", Value: 1}},
					Options: options.Index().SetName("uq_feed_url").SetUnique(true),
				},
				{
					Keys:    bson.D{{Key: "enabled", Value: 1}, {Key: "next_scheduled_at", Value: 1}},
					Options: options.Index().SetName("ix_due_for_collection"),
				},
				{
					Keys: bson.D{
						{Key: "enabled", Value: 1},
						{Key: "priority", Value: -1},
						{Key: "next_scheduled_at", Value: 1},
					},
					Options: options.Index().SetName("ix_due_by_priority"),
				},
				{
					Keys:    bson.D{{Key: "health_status", Value: 1}, {Key: "consecutive_failures", Value: -1}},
					Options: options.Index().SetName("ix_health"),
				},
				{
					Keys: bson.D{
						{Key: "country", Value: 1},
						{Key: "state", Value: 1},
						{Key: "city", Value: 1},
					},
					Options: options.Index().SetName("ix_region"),
				},
				{
					Keys:    bson.D{{Key: "type", Value: 1}, {Key: "enabled", Value: 1}},
					Options: options.Index().SetName("ix_type_enabled"),
				},
			},
		},
		{
			Collection: CollectionArticles,
			Models: []mongo.IndexModel{
				{
					Keys:    bson.D{{Key: "dedup_id", Value: 1}},
					Options: options.Index().SetName("uq_dedup_id").SetUnique(true),
				},
				{
					Keys:    bson.D{{Key: "normalized_url", Value: 1}},
					Options: options.Index().SetName("uq_normalized_url").SetUnique(true),
				},
				{
					Keys: bson.D{{Key: "source_id", Value: 1}, {Key: "feed_guid", Value: 1}},
					Options: options.Index().
						SetName("uq_source_feed_guid").
						SetUnique(true).
						SetPartialFilterExpression(bson.D{{Key: "feed_guid", Value: bson.D{{Key: "$gt", Value: ""}}}}),
				},
				{
					Keys: bson.D{{Key: "canonical_url", Value: 1}},
					Options: options.Index().
						SetName("ix_canonical_url").
						SetPartialFilterExpression(bson.D{{Key: "canonical_url", Value: bson.D{{Key: "$gt", Value: ""}}}}),
				},
				{
					Keys:    bson.D{{Key: "content_hash", Value: 1}},
					Options: options.Index().SetName("ix_content_hash"),
				},
				{
					Keys:    bson.D{{Key: "source_id", Value: 1}, {Key: "published_at", Value: -1}},
					Options: options.Index().SetName("ix_source_published"),
				},
				{
					Keys:    bson.D{{Key: "published_at", Value: -1}, {Key: "_id", Value: -1}},
					Options: options.Index().SetName("ix_published_cursor"),
				},
				{
					Keys:    bson.D{{Key: "collected_at", Value: -1}, {Key: "_id", Value: -1}},
					Options: options.Index().SetName("ix_collected_cursor"),
				},
				{
					Keys:    bson.D{{Key: "language", Value: 1}, {Key: "published_at", Value: -1}},
					Options: options.Index().SetName("ix_language_published"),
				},
				// Region is the axis this whole system is organised around, so
				// filtering by it must not be a collection scan. The prefix
				// order matches how a caller narrows: country, then state,
				// then city.
				{
					Keys: bson.D{
						{Key: "country", Value: 1},
						{Key: "state", Value: 1},
						{Key: "city", Value: 1},
						{Key: "published_at", Value: -1},
					},
					Options: options.Index().SetName("ix_region_published"),
				},
				{
					Keys:    bson.D{{Key: "processing_status", Value: 1}, {Key: "collected_at", Value: -1}},
					Options: options.Index().SetName("ix_status_collected"),
				},
			},
		},
		{
			Collection: CollectionCollectionRuns,
			Models: []mongo.IndexModel{
				{
					Keys:    bson.D{{Key: "source_id", Value: 1}, {Key: "started_at", Value: -1}},
					Options: options.Index().SetName("ix_source_started"),
				},
				{
					Keys:    bson.D{{Key: "started_at", Value: -1}},
					Options: options.Index().SetName("ix_started"),
				},
				{
					Keys:    bson.D{{Key: "status", Value: 1}, {Key: "started_at", Value: -1}},
					Options: options.Index().SetName("ix_status_started"),
				},
			},
		},
		{
			Collection: CollectionFeedCache,
			Models: []mongo.IndexModel{
				{
					Keys:    bson.D{{Key: "source_id", Value: 1}},
					Options: options.Index().SetName("uq_cache_source").SetUnique(true),
				},
			},
		},
		{
			Collection: CollectionLocks,
			Models: []mongo.IndexModel{
				// TTL(0) lets MongoDB reap leases whose expiry has passed, so a
				// crashed collector cannot hold a source lock forever.
				{
					Keys:    bson.D{{Key: "expires_at", Value: 1}},
					Options: options.Index().SetName("ttl_expires_at").SetExpireAfterSeconds(0),
				},
				{
					Keys:    bson.D{{Key: "resource", Value: 1}, {Key: "expires_at", Value: 1}},
					Options: options.Index().SetName("ix_resource_expiry"),
				},
			},
		},
	}
}

// EnsureCollections creates any missing collection and returns the names it
// created. It is safe to run repeatedly.
func EnsureCollections(ctx context.Context, db *mongo.Database) ([]string, error) {
	existing, err := db.ListCollectionNames(ctx, bson.D{})
	if err != nil {
		return nil, fmt.Errorf("mongodb: list collections: %w", err)
	}

	present := make(map[string]struct{}, len(existing))
	for _, name := range existing {
		present[name] = struct{}{}
	}

	var created []string
	for _, name := range Collections() {
		if _, ok := present[name]; ok {
			continue
		}
		if err := db.CreateCollection(ctx, name); err != nil {
			if isNamespaceExists(err) {
				continue
			}
			return created, fmt.Errorf("mongodb: create collection %q: %w", name, err)
		}
		created = append(created, name)
	}
	return created, nil
}

// EnsureIndexes applies the index plan. createIndexes is idempotent for an
// unchanged specification, so reruns are no-ops.
func EnsureIndexes(ctx context.Context, db *mongo.Database) (map[string][]string, error) {
	applied := make(map[string][]string, len(IndexPlan()))

	for _, ci := range IndexPlan() {
		names, err := db.Collection(ci.Collection).Indexes().CreateMany(ctx, ci.Models)
		if err != nil {
			return applied, fmt.Errorf("mongodb: create indexes on %q: %w", ci.Collection, err)
		}
		applied[ci.Collection] = names
	}
	return applied, nil
}

func isNamespaceExists(err error) bool {
	var cmdErr mongo.CommandError
	return errors.As(err, &cmdErr) && cmdErr.Code == namespaceExistsCode
}
