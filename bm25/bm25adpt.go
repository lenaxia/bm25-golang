package bm25

import (
    "errors"
    "log"
)

// BM25Adpt is an implementation of the BM25Adpt variant.
type BM25Adpt struct {
    *bm25Base
    k1    float64
    b     float64
    delta float64
}

// NewBM25Adpt creates a new instance of the BM25Adpt struct.
func NewBM25Adpt(corpus []string, tokenizer func(string) []string, k1 float64, b float64, delta float64, logger *log.Logger) (*BM25Adpt, error) {
    if k1 < 0 {
        return nil, errors.New("k1 must be non-negative")
    }

    if b < 0 || b > 1 {
        return nil, errors.New("b must be between 0 and 1")
    }

    if delta < 0 {
        return nil, errors.New("delta must be non-negative")
    }

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
func (a *BM25Adpt) GetScores(query []string) ([]float64, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    scores := make([]float64, a.corpusSize)
    for _, q := range query {
        qFreq := make([]float64, a.corpusSize)
        for i, doc := range a.corpus {
            docStr := JoinTokens(doc, " ")
            freq, _ := CountTermFreq(q, docStr, a.tokenizer) // Ignore the error for now
            qFreq[i] = float64(freq)
        }

        idf, err := a.IDF(q)
        if err != nil {
            if a.logger != nil {
                a.logger.Printf("Error calculating IDF for term '%s': %v", q, err)
            }
            continue
        }

        for i, docLen := range a.docLengths {
            k := a.k1 * (1 - a.b + a.b*float64(docLen)/a.avgDocLen)
            scores[i] += idf * (a.delta + (qFreq[i] * (1 + k)) / (qFreq[i] + k))
        }
    }

    return scores, nil
}

// GetBatchScores returns the BM25 scores for the given query and a subset of documents.
func (a *BM25Adpt) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
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
            if docID < 0 || docID >= a.corpusSize {
                if a.logger != nil {
                    a.logger.Printf("Invalid document ID: %d", docID)
                }
                continue
            }
            docStr := JoinTokens(a.corpus[docID], " ")
            freq, _ := CountTermFreq(q, docStr, a.tokenizer) // Ignore the error for now
            qFreq[i] = float64(freq)
        }

        idf, err := a.IDF(q)
        if err != nil {
            if a.logger != nil {
                a.logger.Printf("Error calculating IDF for term '%s': %v", q, err)
            }
            continue
        }

        for i, docID := range docIDs {
            if docID < 0 || docID >= a.corpusSize {
                continue
            }
            docLen := a.docLengths[docID]
            k := a.k1 * (1 - a.b + a.b*float64(docLen)/a.avgDocLen)
            scores[i] += idf * (a.delta + (qFreq[i] * (1 + k)) / (qFreq[i] + k))
        }
    }

    return scores, nil
}

// GetTopN returns the top N documents for the given query.
func (a *BM25Adpt) GetTopN(query []string, n int) ([]string, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    if n <= 0 {
        if a.logger != nil {
            a.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
        }
        return []string{}, nil
    }

    scores, err := a.GetScores(query)
    if err != nil {
        return nil, err
    }

    topNIndices, err := TopNIndices(scores, n)
    if err != nil {
        return nil, err
    }

    topDocs := make([]string, len(topNIndices))
    for i, idx := range topNIndices {
        topDocs[i] = JoinTokens(a.corpus[idx], " ")
    }

    return topDocs, nil
}
