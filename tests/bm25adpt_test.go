package bm25

import (
    "log"
    "os"
    "strings"
    "bm25"
    "testing"
)

func TestNewBM25Adpt(t *testing.T) {
	corpus := []string{"hello world", "goodbye world"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	k1 := 1.2
	b := 0.75
	delta := 0.5
	logger := log.New(os.Stderr, "", log.LstdFlags)

	_, err := NewBM25Adpt(corpus, tokenizer, k1, b, delta, logger)
	if err != nil {
		t.Errorf("NewBM25Adpt() failed: %v", err)
	}
}

func TestBM25Adpt_GetScores(t *testing.T) {
	corpus := []string{"hello world", "goodbye world"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	k1 := 1.2
	b := 0.75
	delta := 0.5
	logger := log.New(os.Stderr, "", log.LstdFlags)

	bm25, _ := NewBM25Adpt(corpus, tokenizer, k1, b, delta, logger)
	query := []string{"hello"}
	scores := bm25.GetScores(query)

	if len(scores) != 2 {
		t.Errorf("GetScores() returned incorrect number of scores: %d", len(scores))
	}
}

func TestBM25Adpt_GetBatchScores(t *testing.T) {
	corpus := []string{"hello world", "goodbye world", "another doc"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	k1 := 1.2
	b := 0.75
	delta := 0.5
	logger := log.New(os.Stderr, "", log.LstdFlags)

	bm25, _ := NewBM25Adpt(corpus, tokenizer, k1, b, delta, logger)
	query := []string{"hello"}
	docIDs := []int{0, 2}
	scores := bm25.GetBatchScores(query, docIDs)

	if len(scores) != 2 {
		t.Errorf("GetBatchScores() returned incorrect number of scores: %d", len(scores))
	}
}

func TestBM25Adpt_GetTopN(t *testing.T) {
	corpus := []string{"hello world", "goodbye world", "another doc"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	k1 := 1.2
	b := 0.75
	delta := 0.5
	logger := log.New(os.Stderr, "", log.LstdFlags)

	bm25, _ := NewBM25Adpt(corpus, tokenizer, k1, b, delta, logger)
	query := []string{"hello"}
	topDocs := bm25.GetTopN(query, 2)

	if len(topDocs) != 2 {
		t.Errorf("GetTopN() returned incorrect number of documents: %d", len(topDocs))
	}
}
