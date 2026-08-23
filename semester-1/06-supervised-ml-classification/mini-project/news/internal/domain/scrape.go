package domain

import (
	"errors"
	"strings"
	"time"
)

// ScrapeStatus is the outcome of the enrichment stage's most recent attempt on
// an article. It answers "what happened when we last tried to fetch the full
// text", which is a different question from ProcessingStatus — that one answers
// "how far through the pipeline is this article".
type ScrapeStatus string

// The states an article can be in with respect to full-text enrichment.
//
// Several feeds already carry the whole story (MIT Technology Review ships
// ~1500 words per item), several carry a one-sentence teaser, and a few
// publishers refuse automated clients outright. The statuses below keep those
// three cases apart so a run can tell "nothing to do" from "try again later"
// from "never try again".
const (
	// ScrapeStatusNotNeeded means the feed already supplied the full article.
	ScrapeStatusNotNeeded ScrapeStatus = "not_needed"

	// ScrapeStatusPending means the article is eligible and not yet attempted.
	ScrapeStatusPending ScrapeStatus = "pending"

	// ScrapeStatusScraping means a worker holds a claim on this article. A
	// claim older than the configured TTL is reclaimed: the worker that took it
	// has died.
	ScrapeStatusScraping ScrapeStatus = "scraping"

	// ScrapeStatusSuccess means the fetched article replaced the feed's text.
	ScrapeStatusSuccess ScrapeStatus = "success"

	// ScrapeStatusNoNewContent means the fetch succeeded but produced nothing
	// better than what was already stored. Retrying cannot improve on that.
	ScrapeStatusNoNewContent ScrapeStatus = "no_new_content"

	// ScrapeStatusRobotsDisallowed means the publisher's robots.txt refuses this
	// collector, or could not be read. Terminal by policy, not by failure.
	ScrapeStatusRobotsDisallowed ScrapeStatus = "robots_disallowed"

	// ScrapeStatusBlocked means the server answered, but with a bot wall or a
	// paywall rather than the article. Repeating the identical request cannot
	// change the answer.
	ScrapeStatusBlocked ScrapeStatus = "blocked"

	// ScrapeStatusGone means the article no longer exists — 404, 410, or a
	// redirect that landed somewhere other than the article.
	ScrapeStatusGone ScrapeStatus = "gone"

	// ScrapeStatusFetchFailed means a transient transport problem: a timeout, a
	// 5xx, a DNS failure. Worth another attempt later.
	ScrapeStatusFetchFailed ScrapeStatus = "fetch_failed"

	// ScrapeStatusExtractFailed means the page was fetched but no usable text
	// came out of it. Worth one more attempt in case the page was a transient
	// error screen, but not more: the same markup yields the same result.
	ScrapeStatusExtractFailed ScrapeStatus = "extract_failed"

	// ScrapeStatusFailed means the attempt budget is spent.
	ScrapeStatusFailed ScrapeStatus = "failed"
)

// Enrichment tuning. These are domain floors; the scraper's configuration may
// raise them but not lower them past the point where the result is meaningless.
const (
	// MinScrapedWords is the shortest extraction accepted as an article. It is
	// deliberately low: real news briefs run to a couple of hundred words, and a
	// threshold set for feature-length pieces would reject them. Its job is only
	// to catch nav furniture and bot-check pages, which come in far below it.
	MinScrapedWords = 80

	// FullTextWords is the length above which a feed item is assumed to already
	// hold the whole story, so no fetch is attempted. A teaser is seldom this
	// long, and anything below it may still have more text on the page — the
	// never-regress rule in BetterContent is what makes guessing wrong cheap.
	FullTextWords = 500

	// DefaultMaxScrapeAttempts is how many times one article may be tried before
	// it is abandoned.
	DefaultMaxScrapeAttempts = 3

	// MaxExtractAttempts caps retries after an extraction that produced nothing.
	// Re-parsing unchanged markup produces the same nothing.
	MaxExtractAttempts = 2

	// DefaultScrapeRetryBase is the wait after a first transient failure; it
	// doubles per attempt up to MaxScrapeBackoff.
	DefaultScrapeRetryBase = 15 * time.Minute
	MaxScrapeBackoff       = 6 * time.Hour
)

// Enrichment outcomes a caller may want to recognise. They are errors so a
// scrape can return one path, but only the transport ones are failures: the
// rest are ordinary results that happen to mean "stop".
var (
	// ErrRobotsDisallowed means robots.txt refuses this collector the URL.
	ErrRobotsDisallowed = errors.New("domain: robots.txt disallows this url")

	// ErrScrapeBlocked means a bot wall or paywall answered instead of the article.
	ErrScrapeBlocked = errors.New("domain: request was blocked by the publisher")

	// ErrArticleGone means the article is no longer published at its URL.
	ErrArticleGone = errors.New("domain: article no longer exists")

	// ErrExtractionEmpty means no extractor found usable text on the page.
	ErrExtractionEmpty = errors.New("domain: no article text could be extracted")

	// ErrNoNewContent means the extraction was not an improvement on what is
	// already stored, so the article is left alone.
	ErrNoNewContent = errors.New("domain: scrape produced nothing better than the stored content")
)

// Valid reports whether s is a status this package defines.
func (s ScrapeStatus) Valid() bool {
	switch s {
	case ScrapeStatusNotNeeded, ScrapeStatusPending, ScrapeStatusScraping,
		ScrapeStatusSuccess, ScrapeStatusNoNewContent, ScrapeStatusRobotsDisallowed,
		ScrapeStatusBlocked, ScrapeStatusGone, ScrapeStatusFetchFailed,
		ScrapeStatusExtractFailed, ScrapeStatusFailed:
		return true
	}
	return false
}

// Retryable reports whether an article in this state should be offered to a
// later run. It is the single authority on that question: deciding it at each
// call site is how a permanently broken URL ends up being fetched daily forever.
func (s ScrapeStatus) Retryable() bool {
	switch s {
	case ScrapeStatusPending, ScrapeStatusFetchFailed, ScrapeStatusExtractFailed:
		return true
	}
	return false
}

// HasFullText reports whether the stored content is the whole article, whether
// it arrived in the feed or was fetched afterwards. This is the filter a
// consumer wants when it needs article bodies rather than headlines.
func (s ScrapeStatus) HasFullText() bool {
	return s == ScrapeStatusSuccess || s == ScrapeStatusNotNeeded
}

// ScrapeBackoff returns how long to wait after a failed attempt, doubling per
// attempt and capped. It mirrors the collection backoff on Source so the
// codebase has one retry idiom rather than two.
func ScrapeBackoff(base time.Duration, attempts int) time.Duration {
	if base <= 0 {
		base = DefaultScrapeRetryBase
	}
	d := base
	for i := 1; i < attempts && d < MaxScrapeBackoff; i++ {
		d *= 2
	}
	return min(d, MaxScrapeBackoff)
}

// WordCount counts whitespace-separated tokens. Extraction quality is judged in
// words rather than bytes because byte length varies with markup and encoding
// while the reader's sense of "is this the whole story" does not.
func WordCount(s string) int { return len(strings.Fields(s)) }

// NeedsScrape reports whether an article is worth fetching in full.
//
// The test is generic rather than a list of known-thin publishers: a feed is
// judged only by how much text it actually delivered. Anything shorter than a
// whole story is worth one fetch, because the page may hold more — The Guardian
// sends 96 words and its page yields 427; The Verge sends 139 and yields 395.
// Guessing wrong is cheap: BetterContent refuses to replace good text with bad,
// so a wasted fetch costs a request and nothing else.
//
// This subsumes the teaser shapes seen in the wild, including a publisher that
// repeats its summary as the content: that content is short, so it qualifies.
func NeedsScrape(a *Article) bool {
	if a == nil {
		return false
	}
	return WordCount(a.Content) < FullTextWords
}

// BetterContent reports whether candidate should replace existing.
//
// This is the guard that stops enrichment destroying data. A page whose body is
// rendered client-side yields a few dozen words of navigation furniture, and
// overwriting a good feed article with that is worse than never scraping at all.
func BetterContent(existing, candidate string) bool {
	c := WordCount(candidate)
	if c < MinScrapedWords {
		return false
	}
	return c > WordCount(existing)
}

// ScrapeResult is one finished attempt, in the shape the repository writes. It
// carries every field the update touches so content and status can never be
// written apart.
type ScrapeResult struct {
	Status   ScrapeStatus
	Content  string
	Attempts int
	At       time.Time

	// NextAt is when the article becomes eligible again. It is nil for a
	// terminal status, which is what removes the article from the backlog.
	NextAt *time.Time
}

// ScrapeBacklogFilter selects the articles a run should work on.
type ScrapeBacklogFilter struct {
	// Now bounds NextScrapeAt. It is supplied rather than read from the clock so
	// a test can drive the backoff schedule.
	Now time.Time

	// Limit bounds one run's work.
	Limit int

	// MaxAttempts excludes articles that have spent their budget.
	MaxAttempts int

	// PublishedAfter drops stale articles. Chasing a three-week-old URL mostly
	// buys 404s, and the story is no longer worth having.
	PublishedAfter time.Time
}
