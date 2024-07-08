package bm25

import (
	"testing"
)

func TestNewBM25Okapi(t *testing.T) {
	corpus := []string{"hello world", "goodbye world"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	k1 := 1.2
	b := 0.75
	epsilon := 0.1
	logger := log.New(os.Stderr, "", log.LstdFlags)

	_, err := NewBM25Okapi(corpus, tokenizer, k1, b, epsilon, logger)
	if err != nil {
		t.Errorf("NewBM25Okapi() failed: %v", err)
	}
}

func TestBM25Okapi_GetScores(t *testing.T) {
	corpus := []string{"hello world", "goodbye world"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	k1 := 1.2
	b := 0.75
	epsilon := 0.1
	logger := log.New(os.Stderr, "", log.LstdFlags)

	bm25, _ := NewBM25Okapi(corpus, tokenizer, k1, b, epsilon, logger)
	query := []string{"hello"}
	scores := bm25.GetScores(query)

	if len(scores) != 2 {
		t.Errorf("GetScores() returned incorrect number of scores: %d", len(scores))
	}
}

func TestBM25Okapi_GetBatchScores(t *testing.T) {
	corpus := []string{"hello world", "goodbye world", "another doc"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	k1 := 1.2
	b := 0.75
	epsilon := 0.1
	logger := log.New(os.Stderr, "", log.LstdFlags)

	bm25, _ := NewBM25Okapi(corpus, tokenizer, k1, b, epsilon, logger)
	query := []string{"hello"}
	docIDs := []int{0, 2}
	scores := bm25.GetBatchScores(query, docIDs)

	if len(scores) != 2 {
		t.Errorf("GetBatchScores() returned incorrect number of scores: %d", len(scores))
	}
}

func TestBM25Okapi_GetTopN(t *testing.T) {
	corpus := []string{"hello world", "goodbye world", "another doc"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	k1 := 1.2
	b := 0.75
	epsilon := 0.1
	logger := log.New(os.Stderr, "", log.LstdFlags)

	bm25, _ := NewBM25Okapi(corpus, tokenizer, k1, b, epsilon, logger)
	query := []string{"hello"}
	topDocs := bm25.GetTopN(query, 2)

	if len(topDocs) != 2 {
		t.Errorf("GetTopN() returned incorrect number of documents: %d", len(topDocs))
	}
}
