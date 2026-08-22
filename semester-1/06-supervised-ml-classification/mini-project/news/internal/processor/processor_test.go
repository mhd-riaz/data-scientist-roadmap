package processor

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/riaz/newscollector/internal/domain"
	"github.com/riaz/newscollector/internal/repository"
)

var fixedNow = time.Date(2026, 8, 22, 10, 30, 0, 0, time.UTC)

func fixedClock() time.Time { return fixedNow }

// fakeArticleRepo enforces the same unique keys the articles indexes do, so the
// duplicate paths a test exercises are the ones production takes.
type fakeArticleRepo struct {
	stored []domain.Article

	// createErr and findErr inject a repository failure.
	createErr error
	findErr   error

	// createHook runs before a successful insert, so a test can simulate another
	// collector winning the race between the lookup and the write.
	createHook func()

	creates int
}

var _ repository.ArticleRepository = (*fakeArticleRepo)(nil)

func (f *fakeArticleRepo) Create(_ context.Context, a *domain.Article) error {
	if f.createErr != nil {
		return f.createErr
	}
	if f.createHook != nil {
		f.createHook()
	}
	f.creates++

	for _, existing := range f.stored {
		if existing.DedupID == a.DedupID || existing.NormalizedURL == a.NormalizedURL ||
			(existing.SourceID == a.SourceID && existing.FeedGUID != "" && existing.FeedGUID == a.FeedGUID) {
			return repository.ErrDuplicate
		}
	}
	f.stored = append(f.stored, *a)
	return nil
}

func (f *fakeArticleRepo) FindByIdentity(_ context.Context, id domain.ArticleIdentity) (*domain.Article, error) {
	if f.findErr != nil {
		return nil, f.findErr
	}
	for i, a := range f.stored {
		switch {
		case a.NormalizedURL == id.NormalizedURL,
			id.CanonicalURL != "" && a.CanonicalURL == id.CanonicalURL,
			id.FeedGUID != "" && a.SourceID == id.SourceID && a.FeedGUID == id.FeedGUID,
			id.ContentHash != "" && a.SourceID == id.SourceID && a.ContentHash == id.ContentHash:
			return &f.stored[i], nil
		}
	}
	return nil, repository.ErrNotFound
}

func testSource() *domain.Source {
	return &domain.Source{
		ID:       "018f3f7e-1c2a-7f24-9a3f-8f9f2a0a5c11",
		Name:     "Mysuru Daily",
		FeedURL:  "https://news.example.com/feed.xml",
		Type:     domain.SourceTypeRSS,
		Language: "en",
		Country:  "IN",
		State:    "Karnataka",
		City:     "Mysuru",
	}
}

func item(title, link, guid string) domain.FeedItem {
	published := fixedNow.Add(-2 * time.Hour)
	return domain.FeedItem{Title: title, Link: link, GUID: guid, Summary: "A summary.", PublishedAt: &published}
}

func TestProcessStoresNewArticles(t *testing.T) {
	repo := &fakeArticleRepo{}

	res, err := New(repo, fixedClock).Process(t.Context(), testSource(), []domain.FeedItem{
		item("One", "https://news.example.com/1", "g-1"),
		item("Two", "https://news.example.com/2", "g-2"),
	})
	if err != nil {
		t.Fatalf("Process: %v", err)
	}

	if res != (Result{Items: 2, Stored: 2}) {
		t.Fatalf("Result = %+v, want 2 items stored", res)
	}
	if len(repo.stored) != 2 {
		t.Fatalf("stored %d articles, want 2", len(repo.stored))
	}
	if repo.stored[0].SourceName != "Mysuru Daily" || repo.stored[0].City != "Mysuru" {
		t.Errorf("the source's attribution and region were not carried: %+v", repo.stored[0])
	}
}

func TestProcessSkipsArticlesAlreadyStored(t *testing.T) {
	repo := &fakeArticleRepo{}
	p := New(repo, fixedClock)
	items := []domain.FeedItem{item("One", "https://news.example.com/1", "g-1")}

	if _, err := p.Process(t.Context(), testSource(), items); err != nil {
		t.Fatalf("first Process: %v", err)
	}

	// The same feed, polled again: the entry is unchanged and must not be stored twice.
	res, err := p.Process(t.Context(), testSource(), items)
	if err != nil {
		t.Fatalf("second Process: %v", err)
	}
	if res != (Result{Items: 1, Duplicates: 1}) {
		t.Fatalf("Result = %+v, want the item counted as a duplicate", res)
	}
	if len(repo.stored) != 1 {
		t.Errorf("stored %d articles, want 1", len(repo.stored))
	}
}

func TestProcessDeduplicatesWithinOneBatch(t *testing.T) {
	const permalink = "https://news.example.com/1"

	tests := []struct {
		name  string
		batch []domain.FeedItem
	}{
		{
			name: "same link, one carrying a tracking parameter",
			batch: []domain.FeedItem{
				item("One", "https://news.example.com/1", "g-1"),
				item("One", "https://news.example.com/1?utm_source=twitter", "g-2"),
			},
		},
		{
			name: "different links sharing the publisher's permalink",
			batch: []domain.FeedItem{
				item("One", "https://news.example.com/1", permalink),
				item("One, updated", "https://news.example.com/1-amp", permalink),
			},
		},
		{
			name: "different links sharing the publisher's identifier",
			batch: []domain.FeedItem{
				item("One", "https://news.example.com/1", "g-1"),
				item("One, updated", "https://news.example.com/1-amp", "g-1"),
			},
		},
		{
			name: "same text republished under another link",
			batch: []domain.FeedItem{
				item("One", "https://news.example.com/1", "g-1"),
				item("One", "https://news.example.com/1-amp", "g-2"),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			repo := &fakeArticleRepo{}

			res, err := New(repo, fixedClock).Process(t.Context(), testSource(), tc.batch)
			if err != nil {
				t.Fatalf("Process: %v", err)
			}
			if res.Stored != 1 || res.Duplicates != 1 {
				t.Fatalf("Result = %+v, want 1 stored and 1 duplicate", res)
			}
		})
	}
}

// The content hash is the last key tried, and it only settles a duplicate within
// one source: two sources syndicating the same story are two articles.
func TestProcessKeepsTheSameStoryFromTwoSources(t *testing.T) {
	repo := &fakeArticleRepo{}
	p := New(repo, fixedClock)

	other := testSource()
	other.ID = "018f3f7e-1c2a-7f24-9a3f-8f9f2a0a5c99"
	other.Name = "Deccan Bulletin"

	batch := []domain.FeedItem{item("One", "https://news.example.com/1", "g-1")}
	syndicated := []domain.FeedItem{item("One", "https://bulletin.example.net/1", "b-1")}

	if _, err := p.Process(t.Context(), testSource(), batch); err != nil {
		t.Fatalf("Process: %v", err)
	}
	res, err := p.Process(t.Context(), other, syndicated)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}

	if res.Stored != 1 {
		t.Fatalf("Result = %+v, want the syndicated copy stored", res)
	}
}

func TestProcessCountsUnusableItemsWithoutFailingTheBatch(t *testing.T) {
	repo := &fakeArticleRepo{}

	res, err := New(repo, fixedClock).Process(t.Context(), testSource(), []domain.FeedItem{
		item("No link", "", "g-1"),
		item("", "https://news.example.com/2", "g-2"),
		item("Fine", "https://news.example.com/3", "g-3"),
	})
	if err != nil {
		t.Fatalf("Process: %v", err)
	}

	if res != (Result{Items: 3, Stored: 1, Invalid: 2}) {
		t.Fatalf("Result = %+v, want 1 stored and 2 invalid", res)
	}
}

// Between the lookup and the insert another collector may store the same
// article. The unique index rejects the second write, and that is a duplicate,
// not a failure.
func TestProcessTreatsALostRaceAsADuplicate(t *testing.T) {
	repo := &fakeArticleRepo{}
	repo.createHook = func() {
		if len(repo.stored) == 0 {
			rival, err := domain.NewArticle(*testSource(), item("One", "https://news.example.com/1", "g-1"), fixedNow)
			if err != nil {
				t.Fatalf("build the rival article: %v", err)
			}
			repo.stored = append(repo.stored, *rival)
		}
	}

	res, err := New(repo, fixedClock).Process(t.Context(), testSource(), []domain.FeedItem{
		item("One", "https://news.example.com/1", "g-1"),
	})
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if res != (Result{Items: 1, Duplicates: 1}) {
		t.Fatalf("Result = %+v, want the lost race counted as a duplicate", res)
	}
}

func TestProcessStopsOnARepositoryFailure(t *testing.T) {
	boom := errors.New("mongo is down")

	tests := []struct {
		name string
		repo *fakeArticleRepo
	}{
		{name: "lookup fails", repo: &fakeArticleRepo{findErr: boom}},
		{name: "insert fails", repo: &fakeArticleRepo{createErr: boom}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res, err := New(tc.repo, fixedClock).Process(t.Context(), testSource(), []domain.FeedItem{
				item("One", "https://news.example.com/1", "g-1"),
				item("Two", "https://news.example.com/2", "g-2"),
			})
			if !errors.Is(err, boom) {
				t.Fatalf("Process error = %v, want it to wrap the repository failure", err)
			}
			// The counts so far travel with the error so a run record can still
			// report what got through.
			if res.Items != 2 || res.Stored != 0 {
				t.Errorf("Result = %+v", res)
			}
		})
	}
}

func TestProcessRejectsANilSource(t *testing.T) {
	if _, err := New(&fakeArticleRepo{}, fixedClock).Process(t.Context(), nil, nil); err == nil {
		t.Fatal("expected an error for a nil source")
	}
}

func TestNewDefaultsTheClock(t *testing.T) {
	repo := &fakeArticleRepo{}

	if _, err := New(repo, nil).Process(t.Context(), testSource(), []domain.FeedItem{
		item("One", "https://news.example.com/1", "g-1"),
	}); err != nil {
		t.Fatalf("Process: %v", err)
	}
	if repo.stored[0].CollectedAt.IsZero() {
		t.Error("a nil clock must fall back to the wall clock")
	}
}
