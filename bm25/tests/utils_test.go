package bm25_test

import (
    "testing"

    "lenaxia/bm25_golang/bm25"
)

func TestCountTermFreq(t *testing.T) {
    tokenizer := func(s string) []string { return strings.Split(s, " ") }

    // Test case: Counting term frequency for an empty term
    _, err := bm25.CountTermFreq("", "hello world", tokenizer)
    if err == nil {
        t.Errorf("Expected an error for an empty term, but got nil")
    }

    // Test case: Counting term frequency for an empty document
    _, err = bm25.CountTermFreq("hello", "", tokenizer)
    if err == nil {
        t.Errorf("Expected an error for an empty document, but got nil")
    }

    // Test case: Counting term frequency for a nil tokenizer
    _, err = bm25.CountTermFreq("hello", "hello world", nil)
    if err == nil {
        t.Errorf("Expected an error for a nil tokenizer, but got nil")
    }

    // Test case: Counting term frequency for a term not present in the document
    freq, err := bm25.CountTermFreq("nonexistent", "hello world", tokenizer)
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    if freq != 0 {
        t.Errorf("Expected term frequency 0 for a term not present in the document, but got %d", freq)
    }

    // Test case: Counting term frequency for a term present in the document
    freq, err = bm25.CountTermFreq("hello", "hello world", tokenizer)
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    if freq != 1 {
        t.Errorf("Expected term frequency 1 for the term 'hello', but got %d", freq)
    }
}

func TestTopNIndices(t *testing.T) {
    // Test case: Getting top N indices for n <= 0
    _, err := bm25.TopNIndices([]float64{1.0, 2.0, 3.0}, 0)
    if err == nil {
        t.Errorf("Expected an error for n <= 0, but got nil")
    }

    // Test case: Getting top N indices for n > length of scores
    indices, err := bm25.TopNIndices([]float64{1.0, 2.0, 3.0}, 4)
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    expected := []int{2, 1, 0}
    if len(indices) != len(expected) {
        t.Errorf("Expected %d indices, but got %d", len(expected), len(indices))
    }
    for i, idx := range indices {
        if idx != expected[i] {
            t.Errorf("Expected index %d at position %d, but got %d", expected[i], i, idx)
        }
    }

    // Test case: Getting top N indices for scores with duplicates
    indices, err = bm25.TopNIndices([]float64{1.0, 2.0, 2.0, 3.0}, 3)
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    expected = []int{3, 1, 2}
    if len(indices) != len(expected) {
        t.Errorf("Expected %d indices, but got %d", len(expected), len(indices))
    }
    for i, idx := range indices {
        if idx != expected[i] {
            t.Errorf("Expected index %d at position %d, but got %d", expected[i], i, idx)
        }
    }

    // Test case: Getting top N indices for scores with negative values
    indices, err = bm25.TopNIndices([]float64{-1.0, 2.0, -3.0, 4.0}, 2)
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    expected = []int{3, 1}
    if len(indices) != len(expected) {
        t.Errorf("Expected %d indices, but got %d", len(expected), len(indices))
    }
    for i, idx := range indices {
        if idx != expected[i] {
            t.Errorf("Expected index %d at position %d, but got %d", expected[i], i, idx)
        }
    }
}

func TestJoinTokens(t *testing.T) {
    // Test case: Joining an empty slice
    joined := bm25.JoinTokens([]string{})
    if joined != "" {
        t.Errorf("Expected an empty string, but got '%s'", joined)
    }

    // Test case: Joining a slice with empty strings
    joined = bm25.JoinTokens([]string{"", "", ""})
    if joined != " " {
        t.Errorf("Expected a single space, but got '%s'", joined)
    }

    // Test case: Joining a slice with different separators
    joined = bm25.JoinTokens([]string{"hello", "world"}, "-")
    if joined != "hello-world" {
        t.Errorf("Expected 'hello-world', but got '%s'", joined)
    }

    joined = bm25.JoinTokens([]string{"hello", "world", "test"}, "")
    if joined != "helloworldtest" {
        t.Errorf("Expected 'helloworldtest', but got '%s'", joined)
    }
}

func TestMin(t *testing.T) {
    // Test case: Minimum of positive integers
    min := bm25.Min(5, 10)
    if min != 5 {
        t.Errorf("Expected minimum 5, but got %d", min)
    }

    // Test case: Minimum of negative integers
    min = bm25.Min(-10, -5)
    if min != -10 {
        t.Errorf("Expected minimum -10, but got %d", min)
    }

    // Test case: Minimum of equal integers
    min = bm25.Min(7, 7)
    if min != 7 {
        t.Errorf("Expected minimum 7, but got %d", min)
    }
}
