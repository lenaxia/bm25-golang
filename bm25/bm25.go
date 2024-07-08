package bm25

import (
	"errors"
	"log"
	"math"
)

// BM25 is an interface that defines the common methods for all BM25 variants.
type BM25 interface {
	CorpusSize() int
	AvgDocLen() float64
	DocLengths() []int
	IDF(term string) float64
	GetScores(query []string) []float64
	GetBatchScores(query []string, docIDs []int) []float64
	GetTopN(query []string, n int) []string
}

// bm25Base is a base struct that holds common fields and methods for all BM25 variants.
type bm25Base struct {
	corpus       [][]string
	corpusSize   int
	avgDocLen    float64
	docLengths   []int
	termFreqs    map[string]int
	idfCache     map[string]float64
	tokenizer    func(string) []string
	logger       *log.Logger
}

// NewBM25Base creates a new instance of the bm25Base struct.
func NewBM25Base(corpus []string, tokenizer func(string) []string, logger *log.Logger) (*bm25Base, error) {
	if len(corpus) == 0 {
		return nil, errors.New("corpus cannot be empty")
	}

	if tokenizer == nil {
		return nil, errors.New("tokenizer function cannot be nil")
	}

	base := &bm25Base{
		corpus:     make([][]string, len(corpus)),
		termFreqs:  make(map[string]int),
		idfCache:   make(map[string]float64),
		tokenizer:  tokenizer,
		logger:     logger,
	}

	var totalDocLen int
	for i, doc := range corpus {
		tokens := tokenizer(doc)
		base.corpus[i] = tokens
		base.docLengths = append(base.docLengths, len(tokens))
		totalDocLen += len(tokens)

		for _, token := range tokens {
			base.termFreqs[token]++
		}
	}

	base.corpusSize = len(corpus)
	base.avgDocLen = float64(totalDocLen) / float64(base.corpusSize)

	return base, nil
}

// CorpusSize returns the size of the corpus.
func (b *bm25Base) CorpusSize() int {
	return b.corpusSize
}

// AvgDocLen returns the average document length in the corpus.
func (b *bm25Base) AvgDocLen() float64 {
	return b.avgDocLen
}

// DocLengths returns the lengths of all documents in the corpus.
func (b *bm25Base) DocLengths() []int {
	return b.docLengths
}

// IDF returns the inverse document frequency (IDF) of the given term.
func (b *bm25Base) IDF(term string) float64 {
	if idf, ok := b.idfCache[term]; ok {
		return idf
	}

	termFreq, ok := b.termFreqs[term]
	if !ok {
		b.idfCache[term] = 0.0
		return 0.0
	}

	idf := math.Log(float64(b.corpusSize-termFreq+0.5)) - math.Log(float64(termFreq+0.5))
	b.idfCache[term] = idf

	return idf
}
