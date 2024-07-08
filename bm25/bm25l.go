package bm25

// BM25L is an implementation of the BM25L variant.
type BM25L struct {
	*bm25Base
	k1 float64
	b  float64
}

// NewBM25L creates a new instance of the BM25L struct.
func NewBM25L(corpus []string, tokenizer func(string) []string, k1 float64, b float64, logger *log.Logger) (*BM25L, error) {
	base, err := NewBM25Base(corpus, tokenizer, logger)
	if err != nil {
		return nil, err
	}

	return &BM25L{
		bm25Base: base,
		k1:       k1,
		b:        b,
	}, nil
}

// GetScores returns the BM25 scores for the given query.
func (l *BM25L) GetScores(query []string) []float64 {
	scores := make([]float64, l.corpusSize)
	for _, q := range query {
		qFreq := make([]float64, l.corpusSize)
		for i, doc := range l.corpus {
			qFreq[i] = float64(countTermFreq(q, doc))
		}

		idf := l.IDF(q)
		for i, docLen := range l.docLengths {
			k := l.k1 * (1 - l.b + l.b*float64(docLen)/l.avgDocLen)
			scores[i] += idf * (qFreq[i] / (qFreq[i] + k))
		}
	}

	return scores
}

// GetBatchScores returns the BM25 scores for the given query and a subset of documents.
func (l *BM25L) GetBatchScores(query []string, docIDs []int) []float64 {
	scores := make([]float64, len(docIDs))
	for _, q := range query {
		qFreq := make([]float64, len(docIDs))
		for i, docID := range docIDs {
			qFreq[i] = float64(countTermFreq(q, l.corpus[docID]))
		}

		idf := l.IDF(q)
		for i, docID := range docIDs {
			docLen := l.docLengths[docID]
			k := l.k1 * (1 - l.b + l.b*float64(docLen)/l.avgDocLen)
			scores[i] += idf * (qFreq[i] / (qFreq[i] + k))
		}
	}

	return scores
}

// GetTopN returns the top N documents for the given query.
func (l *BM25L) GetTopN(query []string, n int) []string {
	if n <= 0 {
		l.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
		return []string{}
	}

	scores := l.GetScores(query)
	topNIndices := topNIndices(scores, n)

	topDocs := make([]string, len(topNIndices))
	for i, idx := range topNIndices {
		topDocs[i] = joinTokens(l.corpus[idx])
	}

	return topDocs
}
