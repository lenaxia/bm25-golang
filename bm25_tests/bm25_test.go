package bm25_test

import (
    "testing"

    "bm25-golang/bm25"
)

func TestNewBM25Base(t *testing.T) {
    // Test case: Creating a new bm25Base instance with an empty corpus
    _, err := bm25.NewBM25Base([]string{}, func(s string) []string { return []string{} }, nil)
    if err == nil {
        t.Errorf("Expected an error for an empty corpus, but got nil")
    }

    // Test case: Creating a new bm25Base instance with a nil tokenizer
    _, err = bm25.NewBM25Base([]string{"hello", "world"}, nil, nil)
    if err == nil {
        t.Errorf("Expected an error for a nil tokenizer, but got nil")
    }

    // Test case: Creating a new bm25Base instance with valid inputs
    corpus := []string{"hello world", "this is a test"}
    tokenizer := func(s string) []string { return strings.Split(s, " ") }
    _, err = bm25.NewBM25Base(corpus, tokenizer, nil)
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
}

func TestCorpusSize(t *testing.T) {
    corpus := []string{"hello world", "this is a test"}
    tokenizer := func(s string) []string { return strings.Split(s, " ") }
    base, _ := bm25.NewBM25Base(corpus, tokenizer, nil)

    // Test case: Checking the corpus size
    if base.CorpusSize() != 2 {
        t.Errorf("Expected corpus size 2, but got %d", base.CorpusSize())
    }
}

func TestAvgDocLen(t *testing.T) {
    corpus := []string{"hello world", "this is a test"}
    tokenizer := func(s string) []string { return strings.Split(s, " ") }
    base, _ := bm25.NewBM25Base(corpus, tokenizer, nil)

    // Test case: Checking the average document length
    if base.AvgDocLen() != 3.0 {
        t.Errorf("Expected average document length 3.0, but got %.2f", base.AvgDocLen())
    }
}

func TestDocLengths(t *testing.T) {
    corpus := []string{"hello world", "this is a test"}
    tokenizer := func(s string) []string { return strings.Split(s, " ") }
    base, _ := bm25.NewBM25Base(corpus, tokenizer, nil)

    // Test case: Checking the document lengths
    expected := []int{2, 4}
    docLengths := base.DocLengths()
    if len(docLengths) != len(expected) {
        t.Errorf("Expected %d document lengths, but got %d", len(expected), len(docLengths))
    }
    for i, length := range docLengths {
        if length != expected[i] {
            t.Errorf("Expected document length %d at index %d, but got %d", expected[i], i, length)
        }
    }
}

func TestIDF(t *testing.T) {
    corpus := []string{"hello world", "this is a test"}
    tokenizer := func(s string) []string { return strings.Split(s, " ") }
    base, _ := bm25.NewBM25Base(corpus, tokenizer, nil)

    // Test case: Calculating IDF for an empty term
    _, err := base.IDF("")
    if err == nil {
        t.Errorf("Expected an error for an empty term, but got nil")
    }

    // Test case: Calculating IDF for a term not present in the corpus
    idf, err := base.IDF("nonexistent")
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    if idf != 0.0 {
        t.Errorf("Expected IDF 0.0 for a term not present in the corpus, but got %.2f", idf)
    }

    // Test case: Calculating IDF for a term present in all documents
    idf, err = base.IDF("is")
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    if idf != 0.0 {
        t.Errorf("Expected IDF 0.0 for a term present in all documents, but got %.2f", idf)
    }

    // Test case: Calculating IDF for a term present in some documents
    idf, err = base.IDF("hello")
    if err != nil {
        t.Errorf("Unexpected error: %v", err)
    }
    if idf != 0.69314718055994529 {
        t.Errorf("Expected IDF 0.69314718055994529 for the term 'hello', but got %.2f", idf)
    }
}
