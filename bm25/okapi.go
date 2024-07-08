package bm25

// BM25Okapi is an implementation of the BM25Okapi variant.
type BM25Okapi struct {
	*bm25Base
	k1      float64
	b       float64
	epsilon float64
}

// NewBM25Okapi creates a new instance of the BM25Okapi struct.
func NewBM25Okapi(corpus []string, tokenizer func(string) []string, k1 float64, b float64, epsilon float64, logger *log.Logger) (*BM25Okapi, error) {
	base, err := NewBM25Base(corpus, tokenizer, logger)
	if err != nil {
		return nil, err
	}

	return &BM25Okapi{
		bm25Base: base,
		k1:       k1,
		b:        b,
		epsilon:  epsilon,
	}, nil
}

// GetScores returns the BM25 scores for the given query.
func (o *BM25Okapi) GetScores(query []string) []float64 {
	scores := make([]float64, o.corpusSize)
	for _, q := range query {
		qFreq := make([]float64, o.corpusSize)
		for i, doc := range o.corpus {
			qFreq[i] = float64(countTermFreq(q, doc))
		}

		idf := o.IDF(q)
		for i, docLen := range o.docLengths {
			k := o.k1 * (1 - o.b + o.b*float64(docLen)/o.avgDocLen)
			scores[i] += idf * (qFreq[i] / (qFreq[i] + k))
		}
	}

	return scores
}

// GetBatchScores returns the BM25 scores for the given query and a subset of documents.
func (o *BM25Okapi) GetBatchScores(query []string, docIDs []int) []float64 {
	scores := make([]float64, len(docIDs))
	for _, q := range query {
		qFreq := make([]float64, len(docIDs))
		for i, docID := range docIDs {
			qFreq[i] = float64(countTermFreq(q, o.corpus[docID]))
		}

		idf := o.IDF(q)
		for i, docID := range docIDs {
			docLen := o.docLengths[docID]
			k := o.k1 * (1 - o.b + o.b*float64(docLen)/o.avgDocLen)
			scores[i] += idf * (qFreq[i] / (qFreq[i] + k))
		}
	}

	return scores
}

// GetTopN returns the top N documents for the given query.
func (o *BM25Okapi) GetTopN(query []string, n int) []string {
	if n <= 0 {
		o.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
		return []string{}
	}

	scores := o.GetScores(query)
	topNIndices := topNIndices(scores, n)

	topDocs := make([]string, len(topNIndices))
	for i, idx := range topNIndices {
		topDocs[i] = joinTokens(o.corpus[idx])
	}

	return topDocs
}
