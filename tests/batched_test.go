package bm25

import (
	"testing"
)

func TestGetScoresBatched(t *testing.T) {
	corpus := []string{"hello world", "goodbye world"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	k1 := 1.2
	b := 0.75
	delta := 0.5
	logger := log.New(os.Stderr, "", log.LstdFlags)

	bm25, _ := NewBM25Adpt(corpus, tokenizer, k1, b, delta, logger)
	base, _ := NewBM25Base(corpus, tokenizer, logger)
	query := []string{"hello"}
	scores := base.GetScoresBatched(query, bm25, 1)

	if len(scores) != 2 {
		t.Errorf("GetScoresBatched() returned incorrect number of scores: %d", len(scores))
	}
}

func TestGetBatchScoresBatched(t *testing.T) {
	corpus := []string{"hello world", "goodbye world", "another doc"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	k1 := 1.2
	b := 0.75
	delta := 0.5
	logger := log.New(os.Stderr, "", log.LstdFlags)

	bm25, _ := NewBM25Adpt(corpus, tokenizer, k1, b, delta, logger)
	base, _ := NewBM25Base(corpus, tokenizer, logger)
	query := []string{"hello"}
	docIDs := []int{0, 2}
	scores := base.GetBatchScoresBatched(query, docIDs, bm25, 1)

	if len(scores) != 2 {
		t.Errorf("GetBatchScoresBatched() returned incorrect number of scores: %d", len(scores))
	}
}

func TestGetTopNBatched(t *testing.T) {
	corpus := []string{"hello world", "goodbye world", "another doc"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	k1 := 1.2
	b := 0.75
	delta := 0.5
	logger := log.New(os.Stderr, "", log.LstdFlags)

	bm25, _ := NewBM25Adpt(corpus, tokenizer, k1, b, delta, logger)
	base, _ := NewBM25Base(corpus, tokenizer, logger)
	query := []string{"hello"}
	topDocs := base.GetTopNBatched(query, 2, bm25, 1)

	if len(topDocs) != 2 {
		t.Errorf("GetTopNBatched() returned incorrect number of documents: %d", len(topDocs))
	}
}
