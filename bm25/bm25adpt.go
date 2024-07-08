package bm25

// BM25Adpt is an implementation of the BM25Adpt variant.
type BM25Adpt struct {
	*bm25Base
	k1    float64
	b     float64
	delta float64
}

// NewBM25Adpt creates a new instance of the BM25Adpt struct.
func NewBM25Adpt(corpus []string, tokenizer func(string) []string, k1 float64, b float64, delta float64, logger *log.Logger) (*BM25Adpt, error) {
	base, err := NewBM25Base(corpus, tokenizer, logger)
	if err != nil {
		return nil, err
	}

	return &BM25Adpt{
		bm25Base: base,
		k1:       k1,
		b:        b,
		delta:    delta,
	}, nil
}

// GetScores returns the BM25 scores for the given query.
func (a *BM25Adpt) GetScores(query []string) []float64 {
	scores := make([]float64, a.corpusSize)
	for _, q := range query {
		qFreq := make([]float64, a.corpusSize)
		for i, doc := range a.corpus {
			qFreq[i] = float64(countTermFreq(q, doc))
		}

		idf := a.IDF(q)
		for i, docLen := range a.docLengths {
			k := a.k1 * (1 - a.b + a.b*float64(docLen)/a.avgDocLen)
			scores[i] += idf * (a.delta + (qFreq[i] * (1 + k)) / (qFreq[i] + k))
		}
	}

	return scores
}

// GetBatchScores returns the BM25 scores for the given query and a subset of documents.
func (a *BM25Adpt) GetBatchScores(query []string, docIDs []int) []float64 {
	scores := make([]float64, len(docIDs))
	for _, q := range query {
		qFreq := make([]float64, len(docIDs))
		for i, docID := range docIDs {
			qFreq[i] = float64(countTermFreq(q, a.corpus[docID]))
		}

		idf := a.IDF(q)
		for i, docID := range docIDs {
			docLen := a.docLengths[docID]
			k := a.k1 * (1 - a.b + a.b*float64(docLen)/a.avgDocLen)
			scores[i] += idf * (a.delta + (qFreq[i] * (1 + k)) / (qFreq[i] + k))
		}
	}

	return scores
}

// GetTopN returns the top N documents for the given query.
func (a *BM25Adpt) GetTopN(query []string, n int) []string {
	if n <= 0 {
		a.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
		return []string{}
	}

	scores := a.GetScores(query)
	topNIndices := topNIndices(scores, n)

	topDocs := make([]string, len(topNIndices))
	for i, idx := range topNIndices {
		topDocs[i] = joinTokens(a.corpus[idx])
	}

	return topDocs
}
