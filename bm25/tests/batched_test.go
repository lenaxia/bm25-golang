package bm25_test

import (
    "testing"

    "lenaxia/bm25_golang/bm25"
)

func TestGetScoresBatched(t *testing.T) {
    corpus := []string{"hello world", "this is a test"}
    tokenizer := func(s string) []string { return strings.Split(s, " ") }
    bm25, _ := bm25.NewBM25Okapi(corpus, tokenizer, 1.2, 0.75, nil)

    // Test case: Getting scores in batches for an empty query
    _, err := bm25.GetScoresBatched([]string{}, bm25, 1)
    if err == nil {
        t.Errorf("Expected an error for an empty query, but got nil")
    }

    // Test case: Getting scores in batches with an invalid batch size
    _, err = bm25.GetScoresBatched([]string{"hello"}, bm25, 0)
    if err == nil {
        t.Errorf("Expected an error for an invalid batch size, but got nil")
    }

    // Test case: Getting scores in batches for a single-term query
    scores, err := bm25.GetScoresBatched([]string{"hello"}, bm25, 1)
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    expected := []float64{0.6931471805599453, 0.0}
    if len(scores) != len(expected) {
        t.Errorf("Expected %d scores, but got %d", len(expected), len(scores))
    }
    for i, score := range scores {
        if score != expected[i] {
            t.Errorf("Expected score %.2f at index %d, but got %.2f", expected[i], i, score)
        }
    }

    // Test case: Getting scores in batches for a multi-term query
    scores, err = bm25.GetScoresBatched([]string{"this", "test"}, bm25, 1)
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    expected = []float64{0.0, 1.3862943611198906}
    if len(scores) != len(expected) {
        t.Errorf("Expected %d scores, but got %d", len(expected), len(scores))
    }
    for i, score := range scores {
        if score != expected[i] {
            t.Errorf("Expected score %.2f at index %d, but got %.2f", expected[i], i, score)
        }
    }
}

func TestGetBatchScoresBatched(t *testing.T) {
    corpus := []string{"hello world", "this is a test"}
    tokenizer := func(s string) []string { return strings.Split(s, " ") }
    bm25, _ := bm25.NewBM25Okapi(corpus, tokenizer, 1.2, 0.75, nil)

    // Test case: Getting batch scores in batches for an empty query
    _, err := bm25.GetBatchScoresBatched([]string{}, []int{0, 1}, bm25, 1)
    if err == nil {
        t.Errorf("Expected an error for an empty query, but got nil")
    }

    // Test case: Getting batch scores in batches for an empty document IDs slice
    _, err = bm25.GetBatchScoresBatched([]string{"hello"}, []int{}, bm25, 1)
    if err == nil {
        t.Errorf("Expected an error for an empty document IDs slice, but got nil")
    }

    // Test case: Getting batch scores in batches with an invalid batch size
    _, err = bm25.GetBatchScoresBatched([]string{"hello"}, []int{0, 1}, bm25, 0)
    if err == nil {
        t.Errorf("Expected an error for an invalid batch size, but got nil")
    }

    // Test case: Getting batch scores in batches for invalid document IDs
    _, err = bm25.GetBatchScoresBatched([]string{"hello"}, []int{-1, 2}, bm25, 1)
    if err == nil {
        t.Errorf("Expected an error for invalid document IDs, but got nil")
    }

    // Test case: Getting batch scores in batches for a single-term query
    scores, err := bm25.GetBatchScoresBatched([]string{"hello"}, []int{0}, bm25, 1)
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    expected := []float64{0.6931471805599453}
    if len(scores) != len(expected) {
        t.Errorf("Expected %d scores, but got %d", len(expected), len(scores))
    }
    for i, score := range scores {
        if score != expected[i] {
            t.Errorf("Expected score %.2f at index %d, but got %.2f", expected[i], i, score)
        }
    }

    // Test case: Getting batch scores in batches for a multi-term query
    scores, err = bm25.GetBatchScoresBatched([]string{"this", "test"}, []int{1}, bm25, 1)
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    expected = []float64{1.3862943611198906}
    if len(scores) != len(expected) {
        t.Errorf("Expected %d scores, but got %d", len(expected), len(scores))
    }
    for i, score := range scores {
        if score != expected[i] {
            t.Errorf("Expected score %.2f at index %d, but got %.2f", expected[i], i, score)
        }
    }
}

func TestGetTopNBatched(t *testing.T) {
    corpus := []string{"hello world", "this is a test"}
    tokenizer := func(s string) []string { return strings.Split(s, " ") }
    bm25, _ := bm25.NewBM25Okapi(corpus, tokenizer, 1.2, 0.75, nil)

    // Test case: Getting top N documents in batches for an empty query
    _, err := bm25.GetTopNBatched([]string{}, 2, bm25, 1)
    if err == nil {
        t.Errorf("Expected an error for an empty query, but got nil")
    }

    // Test case: Getting top N documents in batches with n <= 0
    _, err = bm25.GetTopNBatched([]string{"hello"}, 0, bm25, 1)
    if err == nil {
        t.Errorf("Expected an error for n <= 0, but got nil")
    }

    // Test case: Getting top N documents in batches with an invalid batch size
    _, err = bm25.GetTopNBatched([]string{"hello"}, 1, bm25, 0)
    if err == nil {
        t.Errorf("Expected an error for an invalid batch size, but got nil")
    }

    // Test case: Getting top N documents in batches for a single-term query
    topDocs, err := bm25.GetTopNBatched([]string{"hello"}, 1, bm25, 1)
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    expected := []string{"hello world"}
    if len(topDocs) != len(expected) {
        t.Errorf("Expected %d top documents, but got %d", len(expected), len(topDocs))
    }
    for i, doc := range topDocs {
        if doc != expected[i] {
            t.Errorf("Expected document '%s' at index %d, but got '%s'", expected[i], i, doc)
        }
    }

// Test case: Getting top N documents in batches for a multi-term query
    topDocs, err = bm25.GetTopNBatched([]string{"this", "test"}, 1, bm25, 1)
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    expected = []string{"this is a test"}
    if len(topDocs) != len(expected) {
        t.Errorf("Expected %d top documents, but got %d", len(expected), len(topDocs))
    }
    for i, doc := range topDocs {
        if doc != expected[i] {
            t.Errorf("Expected document '%s' at index %d, but got '%s'", expected[i], i, doc)
        }
    }
}
