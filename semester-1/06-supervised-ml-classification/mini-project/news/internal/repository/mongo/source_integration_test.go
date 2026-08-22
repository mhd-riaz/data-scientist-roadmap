//go:build integration

package mongo

import (
	"context"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/riaz/newscollector/internal/domain"
	"github.com/riaz/newscollector/internal/mongodb"
	"github.com/riaz/newscollector/internal/repository"
)

// newTestRepo connects to the MongoDB instance named by NEWS_TEST_MONGO_URI
// (default: the local Docker Compose one) using a database unique to this run,
// and applies the real migration so the tests exercise the production indexes.
func newTestRepo(t *testing.T) *SourceRepository {
	t.Helper()

	uri := os.Getenv("NEWS_TEST_MONGO_URI")
	if uri == "" {
		uri = "mongodb://localhost:27017"
	}

	client, err := mongodb.Connect(mongodb.Settings{
		URI: uri,
		// Nanoseconds keep the name unique per test without the '.' that a
		// fractional-second timestamp would introduce; MongoDB forbids it.
		Database:               fmt.Sprintf("news_it_src_%d", time.Now().UnixNano()),
		AppName:                "news-collector-tests",
		ConnectTimeout:         5 * time.Second,
		ServerSelectionTimeout: 5 * time.Second,
		OperationTimeout:       10 * time.Second,
		MaxPoolSize:            5,
	})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := client.Ping(ctx); err != nil {
		t.Fatalf("MongoDB is not reachable at %s: %v", uri, err)
	}
	if _, err := mongodb.EnsureCollections(ctx, client.Database()); err != nil {
		t.Fatalf("EnsureCollections: %v", err)
	}
	if _, err := mongodb.EnsureIndexes(ctx, client.Database()); err != nil {
		t.Fatalf("EnsureIndexes: %v", err)
	}

	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		if err := client.Database().Drop(cleanupCtx); err != nil {
			t.Errorf("drop test database: %v", err)
		}
		if err := client.Close(cleanupCtx); err != nil {
			t.Errorf("close client: %v", err)
		}
	})

	return NewSourceRepository(client.Database())
}

func testContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func newSource(t *testing.T, feedURL string, mutate ...func(*domain.SourceInput)) *domain.Source {
	t.Helper()
	in := domain.SourceInput{
		Name:     "Test Source",
		FeedURL:  feedURL,
		Type:     domain.SourceTypeRSS,
		Language: "en",
		Country:  "IN",
		State:    "Karnataka",
		City:     "Bengaluru",
	}
	for _, m := range mutate {
		m(&in)
	}
	src, err := domain.NewSource(in, time.Now().UTC())
	if err != nil {
		t.Fatalf("NewSource: %v", err)
	}
	return src
}

func TestCreateAndReadBack(t *testing.T) {
	repo := newTestRepo(t)
	ctx := testContext(t)
	src := newSource(t, "https://example.com/one.rss")

	if err := repo.Create(ctx, src); err != nil {
		t.Fatalf("Create: %v", err)
	}

	got, err := repo.GetByID(ctx, src.ID)
	if err != nil {
		t.Fatalf("GetByID: %v", err)
	}
	if got.ID != src.ID || got.FeedURL != src.FeedURL {
		t.Errorf("read back %+v, want %+v", got, src)
	}
	if got.Country != "IN" || got.Language != "en" {
		t.Errorf("region round-tripped as %q/%q", got.Country, got.Language)
	}
	// Round-tripping the interval must not turn seconds into nanoseconds.
	if got.FetchIntervalSeconds != domain.DefaultFetchIntervalSeconds {
		t.Errorf("fetch_interval_seconds = %d, want %d", got.FetchIntervalSeconds, domain.DefaultFetchIntervalSeconds)
	}
	if got.NextScheduledAt.IsZero() || got.CreatedAt.IsZero() {
		t.Error("timestamps did not survive the round trip")
	}
	// The value the API returned on the write must equal the one a read returns,
	// which only holds if the domain already rounded to BSON's precision.
	if !got.CreatedAt.Equal(src.CreatedAt) {
		t.Errorf("created_at read back as %v, want the written %v", got.CreatedAt, src.CreatedAt)
	}
	if !got.UpdatedAt.Equal(src.UpdatedAt) {
		t.Errorf("updated_at read back as %v, want the written %v", got.UpdatedAt, src.UpdatedAt)
	}
	if !got.NextScheduledAt.Equal(src.NextScheduledAt) {
		t.Errorf("next_scheduled_at read back as %v, want the written %v", got.NextScheduledAt, src.NextScheduledAt)
	}
}

// The unique index, not a prior read, must be what rejects a duplicate.
func TestCreateRejectsADuplicateFeedURL(t *testing.T) {
	repo := newTestRepo(t)
	ctx := testContext(t)

	if err := repo.Create(ctx, newSource(t, "https://example.com/dup.rss")); err != nil {
		t.Fatalf("first Create: %v", err)
	}

	err := repo.Create(ctx, newSource(t, "https://example.com/dup.rss"))
	if !errors.Is(err, repository.ErrDuplicate) {
		t.Fatalf("error = %v, want repository.ErrDuplicate", err)
	}
}

func TestGetByIDReturnsNotFound(t *testing.T) {
	repo := newTestRepo(t)
	ctx := testContext(t)

	_, err := repo.GetByID(ctx, "0198f3d2-0000-7000-8000-00000000dead")
	if !errors.Is(err, repository.ErrNotFound) {
		t.Fatalf("error = %v, want repository.ErrNotFound", err)
	}
}

func TestGetByFeedURL(t *testing.T) {
	repo := newTestRepo(t)
	ctx := testContext(t)
	src := newSource(t, "https://example.com/lookup.rss")
	if err := repo.Create(ctx, src); err != nil {
		t.Fatalf("Create: %v", err)
	}

	got, err := repo.GetByFeedURL(ctx, src.FeedURL)
	if err != nil {
		t.Fatalf("GetByFeedURL: %v", err)
	}
	if got.ID != src.ID {
		t.Errorf("id = %q, want %q", got.ID, src.ID)
	}

	if _, err := repo.GetByFeedURL(ctx, "https://example.com/absent.rss"); !errors.Is(err, repository.ErrNotFound) {
		t.Fatalf("error = %v, want repository.ErrNotFound", err)
	}
}

func TestUpdateReplacesTheStoredDocument(t *testing.T) {
	repo := newTestRepo(t)
	ctx := testContext(t)
	src := newSource(t, "https://example.com/update.rss")
	if err := repo.Create(ctx, src); err != nil {
		t.Fatalf("Create: %v", err)
	}

	src.Enabled = false
	src.Priority = 90
	src.LastError = ""
	if err := repo.Update(ctx, src); err != nil {
		t.Fatalf("Update: %v", err)
	}

	got, err := repo.GetByID(ctx, src.ID)
	if err != nil {
		t.Fatalf("GetByID: %v", err)
	}
	if got.Enabled || got.Priority != 90 {
		t.Errorf("stored %+v, want the replacement applied", got)
	}
}

func TestUpdateReportsAnUnknownID(t *testing.T) {
	repo := newTestRepo(t)
	ctx := testContext(t)
	src := newSource(t, "https://example.com/ghost.rss")

	if err := repo.Update(ctx, src); !errors.Is(err, repository.ErrNotFound) {
		t.Fatalf("error = %v, want repository.ErrNotFound", err)
	}
}

func TestUpdateReportsAColliderFeedURL(t *testing.T) {
	repo := newTestRepo(t)
	ctx := testContext(t)
	first := newSource(t, "https://example.com/a.rss")
	second := newSource(t, "https://example.com/b.rss")
	if err := repo.Create(ctx, first); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := repo.Create(ctx, second); err != nil {
		t.Fatalf("Create: %v", err)
	}

	first.FeedURL = second.FeedURL
	if err := repo.Update(ctx, first); !errors.Is(err, repository.ErrDuplicate) {
		t.Fatalf("error = %v, want repository.ErrDuplicate", err)
	}
}

func TestDelete(t *testing.T) {
	repo := newTestRepo(t)
	ctx := testContext(t)
	src := newSource(t, "https://example.com/delete.rss")
	if err := repo.Create(ctx, src); err != nil {
		t.Fatalf("Create: %v", err)
	}

	if err := repo.Delete(ctx, src.ID); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if err := repo.Delete(ctx, src.ID); !errors.Is(err, repository.ErrNotFound) {
		t.Fatalf("second Delete returned %v, want repository.ErrNotFound", err)
	}
}

func TestListFiltersAndPaginates(t *testing.T) {
	repo := newTestRepo(t)
	ctx := testContext(t)

	seed := []struct {
		url     string
		enabled bool
		country string
		typ     domain.SourceType
	}{
		{"https://example.com/1.rss", true, "IN", domain.SourceTypeRSS},
		{"https://example.com/2.rss", true, "IN", domain.SourceTypeAtom},
		{"https://example.com/3.rss", false, "IN", domain.SourceTypeRSS},
		{"https://example.com/4.rss", true, "US", domain.SourceTypeRSS},
	}
	for _, s := range seed {
		src := newSource(t, s.url, func(in *domain.SourceInput) {
			in.Enabled = &s.enabled
			in.Country = s.country
			in.Type = s.typ
		})
		if err := repo.Create(ctx, src); err != nil {
			t.Fatalf("Create %s: %v", s.url, err)
		}
	}

	enabled := true
	rss := domain.SourceTypeRSS
	tests := []struct {
		name      string
		filter    domain.SourceFilter
		wantTotal int64
	}{
		{"no filter", domain.SourceFilter{Limit: 10}, 4},
		{"enabled only", domain.SourceFilter{Limit: 10, Enabled: &enabled}, 3},
		{"country", domain.SourceFilter{Limit: 10, Country: "IN"}, 3},
		{"type", domain.SourceFilter{Limit: 10, Type: &rss}, 3},
		{"enabled and country", domain.SourceFilter{Limit: 10, Enabled: &enabled, Country: "IN"}, 2},
		{"no match", domain.SourceFilter{Limit: 10, Country: "FR"}, 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			page, err := repo.List(ctx, tc.filter)
			if err != nil {
				t.Fatalf("List: %v", err)
			}
			if page.Total != tc.wantTotal {
				t.Errorf("total = %d, want %d", page.Total, tc.wantTotal)
			}
			if int64(len(page.Items)) != tc.wantTotal {
				t.Errorf("items = %d, want %d", len(page.Items), tc.wantTotal)
			}
		})
	}

	// Paging must be stable: the two pages together must cover every source once.
	seen := make(map[string]bool)
	for offset := 0; offset < 4; offset += 2 {
		page, err := repo.List(ctx, domain.SourceFilter{Limit: 2, Offset: offset})
		if err != nil {
			t.Fatalf("List offset %d: %v", offset, err)
		}
		if len(page.Items) != 2 {
			t.Fatalf("page at offset %d holds %d items, want 2", offset, len(page.Items))
		}
		for _, item := range page.Items {
			if seen[item.ID] {
				t.Errorf("source %q appeared on more than one page", item.ID)
			}
			seen[item.ID] = true
		}
	}
	if len(seen) != 4 {
		t.Errorf("paging covered %d sources, want 4", len(seen))
	}
}

func TestListReturnsAnEmptyPageRatherThanNil(t *testing.T) {
	repo := newTestRepo(t)
	ctx := testContext(t)

	page, err := repo.List(ctx, domain.SourceFilter{Limit: 10})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if page.Items == nil {
		t.Error("items = nil, want an empty slice")
	}
	if page.Total != 0 {
		t.Errorf("total = %d, want 0", page.Total)
	}
}
