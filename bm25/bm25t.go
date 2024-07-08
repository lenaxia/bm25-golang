package bm25

// BM25T is an implementation of the BM25T variant.
type BM25T struct {
	*bm25Base
	k1    float64
	b     float64
	delta float64
}

// NewBM25T creates a new instance of the BM25T struct.
func NewBM25T(corpus []string, tokenizer func(string) []string, k1 float64, b float64, delta float64, logger *log.Logger) (*BM25T, error) {
	base, err := NewBM25Base(corpus, tokenizer, logger)
	if err != nil {
		return nil, err
	}

	return &BM25T{
		bm25Base: base,
		k1:       k1,
		b:        b,
		delta:    delta,
	}, nil
}

// GetScores returns the BM25 scores for the given query.
func (t *BM25T) GetScores(query []string) []float64 {
	scores := make([]float64, t.corpusSize)
	for _, q := range query {
		qFreq := make([]float64, t.corpusSize)
		for i, doc := range t.corpus {
			qFreq[i] = float64(countTermFreq(q, doc))
		}

		idf := t.IDF(q)
		for i, docLen := range t.docLengths {
			k := t.k1 * (1 - t.b + t.b*float64(docLen)/t.avgDocLen)
			scores[i] += idf * (t.delta + (qFreq[i] * (1 + k)) / (qFreq[i] + k))
		}
	}

	return scores
}

// GetBatchScores returns the BM25 scores for the given query and a subset of documents.
func (t *BM25T) GetBatchScores(query []string, docIDs []int) []float64 {
	scores := make([]float64, len(docIDs))
	for _, q := range query {
		qFreq := make([]float64, len(docIDs))
		for i, docID := range docIDs {
			qFreq[i] = float64(countTermFreq(q, t.corpus[docID]))
		}

		idf := t.IDF(q)
		for i, docID := range docIDs {
			docLen := t.docLengths[docID]
			k := t.k1 * (1 - t.b + t.b*float64(docLen)/t.avgDocLen)
			scores[i] += idf * (t.delta + (qFreq[i] * (1 + k)) / (qFreq[i] + k))
		}
	}

	return scores
}

// GetTopN returns the top N documents for the given query.
func (t *BM25T) GetTopN(query []string, n int) []string {
	if n <= 0 {
		t.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
		return []string{}
	}

	scores := t.GetScores(query)
	topNIndices := topNIndices(scores, n)

	topDocs := make([]string, len(topNIndices))
	for i, idx := range topNIndices {
		topDocs[i] = joinTokens(t.corpus[idx])
	}

	return topDocs
}
