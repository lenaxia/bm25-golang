package bm25

import (
    "errors"
    "log"
)

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
    if k1 < 0 {
        return nil, errors.New("k1 must be non-negative")
    }

    if b < 0 || b > 1 {
        return nil, errors.New("b must be between 0 and 1")
    }

    if delta < 0 {
        return nil, errors.New("delta must be non-negative")
    }

    if epsilon < 0 {
        return nil, errors.New("epsilon must be non-negative")
    }

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
func (p *BM25Plus) GetScores(query []string) ([]float64, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    scores := make([]float64, p.corpusSize)
    for _, q := range query {
        qFreq := make([]float64, p.corpusSize)
        for i, doc := range p.corpus {
            docStr := JoinTokens(doc, " ")
            freq, _ := CountTermFreq(q, docStr, p.tokenizer) // Ignore the error for now
            qFreq[i] = float64(freq)
        }

        idf, err := p.IDF(q)
        if err != nil {
            if p.logger != nil {
                p.logger.Printf("Error calculating IDF for term '%s': %v", q, err)
            }
            continue
        }

        for i, docLen := range p.docLengths {
            k := p.k1 * (1 - p.b + p.b*float64(docLen)/p.avgDocLen)
            scores[i] += idf * (p.delta + (qFreq[i] / (qFreq[i] + k)))
        }
    }

    return scores, nil
}

// GetBatchScores returns the BM25 scores for the given query and a subset of documents.
func (p *BM25Plus) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    if len(docIDs) == 0 {
        return nil, errors.New("document IDs cannot be empty")
    }

    scores := make([]float64, len(docIDs))
    for _, q := range query {
        qFreq := make([]float64, len(docIDs))
        for i, docID := range docIDs {
            if docID < 0 || docID >= p.corpusSize {
                if p.logger != nil {
                    p.logger.Printf("Invalid document ID: %d", docID)
                }
                continue
            }
            docStr := JoinTokens(p.corpus[docID], " ")
            freq, _ := CountTermFreq(q, docStr, p.tokenizer) // Ignore the error for now
            qFreq[i] = float64(freq)
        }

        idf, err := p.IDF(q)
        if err != nil {
            if p.logger != nil {
                p.logger.Printf("Error calculating IDF for term '%s': %v", q, err)
            }
            continue
        }

        for i, docID := range docIDs {
            if docID < 0 || docID >= p.corpusSize {
                continue
            }
            docLen := p.docLengths[docID]
            k := p.k1 * (1 - p.b + p.b*float64(docLen)/p.avgDocLen)
            scores[i] += idf * (p.delta + (qFreq[i] / (qFreq[i] + k)))
        }
    }

    return scores, nil
}

// GetTopN returns the top N documents for the given query.
func (p *BM25Plus) GetTopN(query []string, n int) ([]string, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    if n <= 0 {
        if p.logger != nil {
            p.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
        }
        return []string{}, nil
    }

    scores, err := p.GetScores(query)
    if err != nil {
        return nil, err
    }

    topNIndices, err := TopNIndices(scores, n)
    if err != nil {
        return nil, err
    }

    topDocs := make([]string, len(topNIndices))
    for i, idx := range topNIndices {
        topDocs[i] = JoinTokens(p.corpus[idx], " ")
    }

    return topDocs, nil
}
