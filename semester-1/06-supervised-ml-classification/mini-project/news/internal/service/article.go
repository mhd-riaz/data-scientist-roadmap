package service

import (
	"context"

	"github.com/riaz/newscollector/internal/domain"
	"github.com/riaz/newscollector/internal/repository"
)

// ArticleService reads collected articles. Articles are written only by the
// collection pipeline, so this service is deliberately read-only: there is no
// use case in Phase 1 for editing or deleting one through an API.
type ArticleService struct {
	repo repository.ArticleRepository
}

// NewArticleService wires the service.
func NewArticleService(repo repository.ArticleRepository) *ArticleService {
	return &ArticleService{repo: repo}
}

// List returns the page of articles matching filter, newest first.
func (s *ArticleService) List(ctx context.Context, filter domain.ArticleFilter) (domain.ArticlePage, error) {
	filter.Normalize()
	if err := filter.Validate(); err != nil {
		return domain.ArticlePage{}, err
	}

	page, err := s.repo.List(ctx, filter)
	if err != nil {
		return domain.ArticlePage{}, translate(err)
	}
	return page, nil
}

// Get returns one article in full.
func (s *ArticleService) Get(ctx context.Context, id string) (*domain.Article, error) {
	if err := domain.ValidateID(id); err != nil {
		return nil, err
	}

	article, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return nil, translate(err)
	}
	return article, nil
}
