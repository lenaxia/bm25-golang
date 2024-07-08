package bm25

import "sync"

// Parallel implementation of BM25 score calculation

// GetScoresParallel returns the BM25 scores for the given query using parallel computation.
func (b *bm25Base) GetScoresParallel(query []string, bm25 BM25) []float64 {
	var wg sync.WaitGroup
	scores := make([]float64, b.corpusSize)
	wg.Add(len(query))

	for _, q := range query {
		go func(q string) {
			defer wg.Done()
			qFreq := make([]float64, b.corpusSize)
			for i, doc := range b.corpus {
				qFreq[i] = float64(countTermFreq(q, doc))
			}

			idf := b.IDF(q)
			for i, docLen := range b.docLengths {
				k := computeK(bm25, docLen)
				scores[i] += idf * computeScore(bm25, qFreq[i], k)
			}
		}(q)
	}

	wg.Wait()
	return scores
}

// computeK computes the k value based on the BM25 variant and document length.
func computeK(bm25 BM25, docLen int) float64 {
	switch bm25 := bm25.(type) {
	case *BM25Okapi:
		return bm25.k1 * (1 - bm25.b + bm25.b*float64(docLen)/bm25.avgDocLen)
	case *BM25L:
		return bm25.k1 * (1 - bm25.b + bm25.b*float64(docLen)/bm25.avgDocLen)
	case *BM25Plus:
		return bm25.k1 * (1 - bm25.b + bm25.b*float64(docLen)/bm25.avgDocLen)
	case *BM25Adpt:
		return bm25.k1 * (1 - bm25.b + bm25.b*float64(docLen)/bm25.avgDocLen)
	case *BM25T:
		return bm25.k1 * (1 - bm25.b + bm25.b*float64(docLen)/bm25.avgDocLen)
	default:
		return 0
	}
}

// computeScore computes the BM25 score based on the BM25 variant and query term frequency.
func computeScore(bm25 BM25, qFreq, k float64) float64 {
	switch bm25 := bm25.(type) {
	case *BM25Okapi:
		return qFreq / (qFreq + k)
	case *BM25L:
		return qFreq / (qFreq + k)
	case *BM25Plus:
		return bm25.delta + (qFreq / (qFreq + k))
	case *BM25Adpt:
		return bm25.delta + (qFreq * (1 + k)) / (qFreq + k)
	case *BM25T:
		return bm25.delta + (qFreq * (1 + k)) / (qFreq + k)
	default:
		return 0
	}
}

// GetBatchScoresParallel returns the BM25 scores for the given query and a subset of documents using parallel computation.
func (b *bm25Base) GetBatchScoresParallel(query []string, docIDs []int, bm25 BM25) []float64 {
	var wg sync.WaitGroup
	scores := make([]float64, len(docIDs))
	wg.Add(len(query))

	for _, q := range query {
		go func(q string) {
			defer wg.Done()
			qFreq := make([]float64, len(docIDs))
			for i, docID := range docIDs {
				qFreq[i] = float64(countTermFreq(q, b.corpus[docID]))
			}

			idf := b.IDF(q)
			for i, docID := range docIDs {
				docLen := b.docLengths[docID]
				k := computeK(bm25, docLen)
				scores[i] += idf * computeScore(bm25, qFreq[i], k)
			}
		}(q)
	}

	wg.Wait()
	return scores
}

// GetTopNParallel returns the top N documents for the given query using parallel computation.
func (b *bm25Base) GetTopNParallel(query []string, n int, bm25 BM25) []string {
	if n <= 0 {
		b.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
		return []string{}
	}

	scores := b.GetScoresParallel(query, bm25)
	topNIndices := topNIndices(scores, n)

	topDocs := make([]string, len(topNIndices))
	for i, idx := range topNIndices {
		topDocs[i] = joinTokens(b.corpus[idx])
	}

	return topDocs
}
