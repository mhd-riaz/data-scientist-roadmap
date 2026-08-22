package main

import (
	"os"
	"strings"
	"testing"

	"github.com/riaz/newscollector/internal/domain"
)

const oneValidSource = `
sources:
  - name: The Hindu — Bengaluru
    feed_url: https://www.thehindu.com/news/cities/bangalore/feeder/default.rss
    type: rss
    language: en
    country: IN
    state: Karnataka
    city: Bengaluru
`

func TestParseSeedFile(t *testing.T) {
	seed, err := parseSeedFile(strings.NewReader(oneValidSource))
	if err != nil {
		t.Fatalf("parseSeedFile: %v", err)
	}
	if len(seed.Sources) != 1 {
		t.Fatalf("parsed %d sources, want 1", len(seed.Sources))
	}

	in := seed.Sources[0].toInput()
	if in.Type != domain.SourceTypeRSS || in.Country != "IN" || in.City != "Bengaluru" {
		t.Errorf("input = %+v, want the file values carried through", in)
	}
	// Omitted optional keys must stay nil so a re-seed cannot reset a tuned value.
	if in.Enabled != nil || in.Priority != nil || in.FetchIntervalSeconds != nil {
		t.Errorf("optional fields = %v/%v/%v, want nil", in.Enabled, in.Priority, in.FetchIntervalSeconds)
	}
}

func TestParseSeedFileRejectsUnknownKeys(t *testing.T) {
	_, err := parseSeedFile(strings.NewReader(`
sources:
  - name: X
    feed_ur1: https://example.com/f.rss
`))
	if err == nil {
		t.Fatal("parseSeedFile accepted a typo'd key, want it rejected")
	}
	if !strings.Contains(err.Error(), "feed_ur1") {
		t.Errorf("error = %v, want it to name the offending key", err)
	}
}

func TestParseSeedFileAcceptsAnEmptyDocument(t *testing.T) {
	seed, err := parseSeedFile(strings.NewReader(""))
	if err != nil {
		t.Fatalf("parseSeedFile: %v", err)
	}
	if len(seed.Sources) != 0 {
		t.Errorf("parsed %d sources, want 0", len(seed.Sources))
	}
}

func TestParseSeedFileRejectsMalformedYAML(t *testing.T) {
	if _, err := parseSeedFile(strings.NewReader("sources: [{")); err == nil {
		t.Fatal("parseSeedFile accepted malformed YAML")
	}
}

func TestValidateAllAcceptsAValidFile(t *testing.T) {
	seed, err := parseSeedFile(strings.NewReader(oneValidSource))
	if err != nil {
		t.Fatalf("parseSeedFile: %v", err)
	}

	inputs, err := validateAll(seed.Sources)
	if err != nil {
		t.Fatalf("validateAll: %v", err)
	}
	if len(inputs) != 1 {
		t.Errorf("validated %d inputs, want 1", len(inputs))
	}
}

func TestValidateAllReportsThePositionOfEveryBadEntry(t *testing.T) {
	seed, err := parseSeedFile(strings.NewReader(`
sources:
  - name: Good
    feed_url: https://example.com/good.rss
    type: rss
    language: en
    country: IN
  - name: Bad scheme
    feed_url: file:///etc/passwd
    type: rss
    language: en
    country: IN
  - name: Bad type
    feed_url: https://example.com/other.rss
    type: json
    language: en
    country: IN
`))
	if err != nil {
		t.Fatalf("parseSeedFile: %v", err)
	}

	_, err = validateAll(seed.Sources)
	if err == nil {
		t.Fatal("validateAll accepted two invalid entries")
	}
	for _, want := range []string{"sources[1]", "sources[2]", "Bad scheme", "Bad type"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err.Error(), want)
		}
	}
}

// A duplicate inside the file would otherwise surface as a confusing conflict
// only once it reached the unique index.
func TestValidateAllRejectsADuplicateFeedURLWithinTheFile(t *testing.T) {
	seed, err := parseSeedFile(strings.NewReader(`
sources:
  - name: First
    feed_url: https://example.com/same.rss
    type: rss
    language: en
    country: IN
  - name: Second
    feed_url: https://example.com/same.rss
    type: atom
    language: en
    country: IN
`))
	if err != nil {
		t.Fatalf("parseSeedFile: %v", err)
	}

	_, err = validateAll(seed.Sources)
	if err == nil {
		t.Fatal("validateAll accepted a feed URL declared twice")
	}
	if !strings.Contains(err.Error(), "already declared at sources[0]") {
		t.Errorf("error = %v, want it to point at the first declaration", err)
	}
}

// The shipped seed file must always be valid; a broken default would fail only
// at deploy time.
func TestShippedSeedFileIsValid(t *testing.T) {
	f, err := os.Open("../../configs/sources.yaml")
	if err != nil {
		t.Fatalf("open shipped seed file: %v", err)
	}
	defer func() { _ = f.Close() }()

	seed, err := parseSeedFile(f)
	if err != nil {
		t.Fatalf("parse shipped seed file: %v", err)
	}
	if len(seed.Sources) == 0 {
		t.Fatal("the shipped seed file declares no sources")
	}
	if _, err := validateAll(seed.Sources); err != nil {
		t.Fatalf("the shipped seed file is invalid: %v", err)
	}
}
