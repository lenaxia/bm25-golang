package bm25

// BM25Plus is an implementation of the BM25Plus variant.
type BM25Plus struct {
	*bm25Base
	k1      float64
	b       float64
	delta   float64
	epsilon float64
}

// NewBM25Plus creates a new instance of the BM25Plus struct.
func NewBM25Plus(corpus []string, tokenizer func(string) []string, k1 float64, b float64, delta float64, epsilon float64, logger *log.Logger) (*BM25Plus, error) {
	base, err := NewBM25Base(corpus, tokenizer, logger)
	if err != nil {
		return nil, err
	}

	return &BM25Plus{
		bm25Base: base,
		k1:       k1,
		b:        b,
		delta:    delta,
		epsilon:  epsilon,
	}, nil
}

// GetScores returns the BM25 scores for the given query.
func (p *BM25Plus) GetScores(query []string) []float64 {
	scores := make([]float64, p.corpusSize)
	for _, q := range query {
		qFreq := make([]float64, p.corpusSize)
		for i, doc := range p.corpus {
			qFreq[i] = float64(countTermFreq(q, doc))
		}

		idf := p.IDF(q)
		for i, docLen := range p.docLengths {
			k := p.k1 * (1 - p.b + p.b*float64(docLen)/p.avgDocLen)
			scores[i] += idf * (p.delta + (qFreq[i] / (qFreq[i] + k)))
		}
	}

	return scores
}

// GetBatchScores returns the BM25 scores for the given query and a subset of documents.
func (p *BM25Plus) GetBatchScores(query []string, docIDs []int) []float64 {
	scores := make([]float64, len(docIDs))
	for _, q := range query {
		qFreq := make([]float64, len(docIDs))
		for i, docID := range docIDs {
			qFreq[i] = float64(countTermFreq(q, p.corpus[docID]))
		}

		idf := p.IDF(q)
		for i, docID := range docIDs {
			docLen := p.docLengths[docID]
			k := p.k1 * (1 - p.b + p.b*float64(docLen)/p.avgDocLen)
			scores[i] += idf * (p.delta + (qFreq[i] / (qFreq[i] + k)))
		}
	}

	return scores
}

// GetTopN returns the top N documents for the given query.
func (p *BM25Plus) GetTopN(query []string, n int) []string {
	if n <= 0 {
		p.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
		return []string{}
	}

	scores := p.GetScores(query)
	topNIndices := topNIndices(scores, n)

	topDocs := make([]string, len(topNIndices))
	for i, idx := range topNIndices {
		topDocs[i] = joinTokens(p.corpus[idx])
	}

	return topDocs
}
