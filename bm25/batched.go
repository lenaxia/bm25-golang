package bm25

// Parallel implementation of BM25 score calculation with batching

// GetScoresBatched returns the BM25 scores for the given query using parallel computation with batching.
func (b *bm25Base) GetScoresBatched(query []string, bm25 BM25, batchSize int) []float64 {
	var wg sync.WaitGroup
	scores := make([]float64, b.corpusSize)
	numBatches := (b.corpusSize + batchSize - 1) / batchSize
	wg.Add(numBatches)

	for i := 0; i < numBatches; i++ {
		start := i * batchSize
		end := min(start+batchSize, b.corpusSize)
		go func(start, end int) {
			defer wg.Done()
			for _, q := range query {
				qFreq := make([]float64, end-start)
				for j, idx := range b.docIndices[start:end] {
					qFreq[j] = float64(countTermFreq(q, b.corpus[idx]))
				}

				idf := b.IDF(q)
				for j, idx := range b.docIndices[start:end] {
					docLen := b.docLengths[idx]
					k := computeK(bm25, docLen)
					scores[idx] += idf * computeScore(bm25, qFreq[j], k)
				}
			}
		}(start, end)
	}

	wg.Wait()
	return scores
}

// GetBatchScoresBatched returns the BM25 scores for the given query and a subset of documents using parallel computation with batching.
func (b *bm25Base) GetBatchScoresBatched(query []string, docIDs []int, bm25 BM25, batchSize int) []float64 {
	var wg sync.WaitGroup
	scores := make([]float64, len(docIDs))
	numBatches := (len(docIDs) + batchSize - 1) / batchSize
	wg.Add(numBatches)

	for i := 0; i < numBatches; i++ {
		start := i * batchSize
		end := min(start+batchSize, len(docIDs))
		go func(start, end int) {
			defer wg.Done()
			for _, q := range query {
				qFreq := make([]float64, end-start)
				for j, docID := range docIDs[start:end] {
					qFreq[j] = float64(countTermFreq(q, b.corpus[docID]))
				}

				idf := b.IDF(q)
				for j, docID := range docIDs[start:end] {
					docLen := b.docLengths[docID]
					k := computeK(bm25, docLen)
					scores[docID] += idf * computeScore(bm25, qFreq[j], k)
				}
			}
		}(start, end)
	}

	wg.Wait()
	return scores
}

// GetTopNBatched returns the top N documents for the given query using parallel computation with batching.
func (b *bm25Base) GetTopNBatched(query []string, n int, bm25 BM25, batchSize int) []string {
	if n <= 0 {
		b.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
		return []string{}
	}

	scores := b.GetScoresBatched(query, bm25, batchSize)
	topNIndices := topNIndices(scores, n)

	topDocs := make([]string, len(topNIndices))
	for i, idx := range topNIndices {
		topDocs[i] = joinTokens(b.corpus[idx])
	}

	return topDocs
}
