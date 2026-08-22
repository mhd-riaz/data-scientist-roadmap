package service

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/riaz/newscollector/internal/domain"
	"github.com/riaz/newscollector/internal/repository"
)

// fakeArticleReadRepo records what the service asked for.
type fakeArticleReadRepo struct {
	article  *domain.Article
	page     domain.ArticlePage
	err      error
	lastFltr domain.ArticleFilter
	calls    int
}

var _ repository.ArticleRepository = (*fakeArticleReadRepo)(nil)

func (f *fakeArticleReadRepo) Create(context.Context, *domain.Article) error { return nil }

func (f *fakeArticleReadRepo) FindByIdentity(context.Context, domain.ArticleIdentity) (*domain.Article, error) {
	return nil, repository.ErrNotFound
}

func (f *fakeArticleReadRepo) GetByID(_ context.Context, _ string) (*domain.Article, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	if f.article == nil {
		return nil, repository.ErrNotFound
	}
	return f.article, nil
}

func (f *fakeArticleReadRepo) List(_ context.Context, filter domain.ArticleFilter) (domain.ArticlePage, error) {
	f.calls++
	f.lastFltr = filter
	return f.page, f.err
}

func TestArticleListNormalizesBeforeQuerying(t *testing.T) {
	repo := &fakeArticleReadRepo{}

	if _, err := NewArticleService(repo).List(t.Context(), domain.ArticleFilter{Country: " in "}); err != nil {
		t.Fatalf("List: %v", err)
	}

	got := repo.lastFltr
	if got.Country != "IN" || got.Limit != domain.DefaultListLimit || got.Sort != domain.SortPublishedAt {
		t.Fatalf("filter = %+v, want it normalised before it reached the repository", got)
	}
}

func TestArticleListRejectsABadFilter(t *testing.T) {
	repo := &fakeArticleReadRepo{}

	_, err := NewArticleService(repo).List(t.Context(), domain.ArticleFilter{Limit: domain.MaxListLimit + 1})

	if !errors.Is(err, domain.ErrValidation) {
		t.Fatalf("error = %v, want a validation error", err)
	}
	if repo.calls != 0 {
		t.Errorf("repository was called %d times, want 0", repo.calls)
	}
}

func TestArticleListReturnsThePage(t *testing.T) {
	repo := &fakeArticleReadRepo{page: domain.ArticlePage{
		Items:      []domain.Article{{ID: "0198f3d2-3333-7000-8000-000000000001"}},
		Limit:      50,
		HasMore:    true,
		NextCursor: "abc",
	}}

	page, err := NewArticleService(repo).List(t.Context(), domain.ArticleFilter{})
	if err != nil {
		t.Fatalf("List: %v", err)
	}

	if len(page.Items) != 1 || !page.HasMore || page.NextCursor != "abc" {
		t.Fatalf("page = %+v, want the repository's page passed through", page)
	}
}

func TestArticleListReportsAStorageFailure(t *testing.T) {
	repo := &fakeArticleReadRepo{err: errors.New("mongo: find articles: timeout")}

	if _, err := NewArticleService(repo).List(t.Context(), domain.ArticleFilter{}); err == nil {
		t.Fatal("List hid a storage failure")
	}
}

func TestArticleGetRejectsAnIdentifierThatIsNotAUUID(t *testing.T) {
	repo := &fakeArticleReadRepo{}

	_, err := NewArticleService(repo).Get(t.Context(), "../../../etc/passwd")

	if !errors.Is(err, domain.ErrValidation) {
		t.Fatalf("error = %v, want a validation error", err)
	}
	if repo.calls != 0 {
		t.Errorf("a malformed identifier reached the repository")
	}
}

func TestArticleGetTranslatesAMissingArticle(t *testing.T) {
	_, err := NewArticleService(&fakeArticleReadRepo{}).Get(t.Context(), "0198f3d2-3333-7000-8000-000000000009")

	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("error = %v, want ErrNotFound", err)
	}
}

func TestArticleGetReturnsTheArticle(t *testing.T) {
	want := &domain.Article{
		ID:          "0198f3d2-3333-7000-8000-000000000001",
		Title:       "A story",
		Content:     "The full text.",
		PublishedAt: time.Now().UTC(),
	}

	got, err := NewArticleService(&fakeArticleReadRepo{article: want}).Get(t.Context(), want.ID)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}

	if got.ID != want.ID || got.Content != want.Content {
		t.Fatalf("article = %+v, want %+v", got, want)
	}
}
