package bm25

import (
	"log"
	"reflect"
	"strings"
	"testing"
)

func TestNewBM25Base(t *testing.T) {
	corpus := []string{"this is a test", "another test document"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	logger := log.New(nil, "", 0)

	// Test with valid input
	base, err := NewBM25Base(corpus, tokenizer, logger)
	if err != nil {
		t.Errorf("NewBM25Base returned an error: %v", err)
	}

	if base.corpusSize != 2 {
		t.Errorf("Incorrect corpus size. Expected: 2, Got: %d", base.corpusSize)
	}

	if base.avgDocLen != 3.5 {
		t.Errorf("Incorrect average document length. Expected: 3.5, Got: %.1f", base.avgDocLen)
	}

	if !reflect.DeepEqual(base.docLengths, []int{4, 3}) {
		t.Errorf("Incorrect document lengths. Expected: [4, 3], Got: %v", base.docLengths)
	}

	// Test with empty corpus
	_, err = NewBM25Base([]string{}, tokenizer, logger)
	if err == nil {
		t.Error("NewBM25Base should return an error for an empty corpus")
	}

	// Test with nil tokenizer
	_, err = NewBM25Base(corpus, nil, logger)
	if err == nil {
		t.Error("NewBM25Base should return an error for a nil tokenizer")
	}
}

func TestCorpusSize(t *testing.T) {
	corpus := []string{"this is a test", "another test document"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	logger := log.New(nil, "", 0)

	base, _ := NewBM25Base(corpus, tokenizer, logger)

	if base.CorpusSize() != 2 {
		t.Errorf("Incorrect corpus size. Expected: 2, Got: %d", base.CorpusSize())
	}
}

func TestAvgDocLen(t *testing.T) {
	corpus := []string{"this is a test", "another test document"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	logger := log.New(nil, "", 0)

	base, _ := NewBM25Base(corpus, tokenizer, logger)

	if base.AvgDocLen() != 3.5 {
		t.Errorf("Incorrect average document length. Expected: 3.5, Got: %.1f", base.AvgDocLen())
	}
}

func TestDocLengths(t *testing.T) {
	corpus := []string{"this is a test", "another test document"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	logger := log.New(nil, "", 0)

	base, _ := NewBM25Base(corpus, tokenizer, logger)

	if !reflect.DeepEqual(base.DocLengths(), []int{4, 3}) {
		t.Errorf("Incorrect document lengths. Expected: [4, 3], Got: %v", base.DocLengths())
	}
}

func TestIDF(t *testing.T) {
	corpus := []string{"this is a test", "another test document"}
	tokenizer := func(s string) []string {
		return strings.Split(s, " ")
	}
	logger := log.New(nil, "", 0)

	base, _ := NewBM25Base(corpus, tokenizer, logger)

	// Test with existing term
	idf := base.IDF("test")
	expected := 0.2876820724517809
	if idf != expected {
		t.Errorf("Incorrect IDF for 'test'. Expected: %.10f, Got: %.10f", expected, idf)
	}

	// Test with non-existing term
	idf = base.IDF("nonexistent")
	if idf != 0.0 {
		t.Errorf("Incorrect IDF for 'nonexistent'. Expected: 0.0, Got: %.10f", idf)
	}
}
