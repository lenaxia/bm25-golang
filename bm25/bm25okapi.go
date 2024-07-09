package bm25

import (
    "errors"
    "log"
)

// BM25Okapi is an implementation of the Okapi BM25 variant.
type BM25Okapi struct {
    *bm25Base
    k1 float64
    b  float64
}

// NewBM25Okapi creates a new instance of the BM25Okapi struct.
func NewBM25Okapi(corpus []string, tokenizer func(string) []string, k1 float64, b float64, logger *log.Logger) (*BM25Okapi, error) {
    if k1 < 0 {
        return nil, errors.New("k1 must be non-negative")
    }

    if b < 0 || b > 1 {
        return nil, errors.New("b must be between 0 and 1")
    }

    base, err := NewBM25Base(corpus, tokenizer, logger)
    if err != nil {
        return nil, err
    }

    return &BM25Okapi{
        bm25Base: base,
        k1:       k1,
        b:        b,
    }, nil
}

// GetScores returns the BM25 scores for the given query.
func (o *BM25Okapi) GetScores(query []string) ([]float64, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    scores := make([]float64, o.corpusSize)
    for _, q := range query {
        qFreq := make([]float64, o.corpusSize)
        for i, doc := range o.corpus {
            qFreq[i] = float64(countTermFreq(q, doc))
        }

        idf, err := o.IDF(q)
        if err != nil {
            if o.logger != nil {
                o.logger.Printf("Error calculating IDF for term '%s': %v", q, err)
            }
            continue
        }

        for i, docLen := range o.docLengths {
            k := o.k1 * (1 - o.b + o.b*float64(docLen)/o.avgDocLen)
            scores[i] += idf * (qFreq[i] / (qFreq[i] + k))
        }
    }

    return scores, nil
}

// GetBatchScores returns the BM25 scores for the given query and a subset of documents.
func (o *BM25Okapi) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
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
            if docID < 0 || docID >= o.corpusSize {
                if o.logger != nil {
                    o.logger.Printf("Invalid document ID: %d", docID)
                }
                continue
            }
            qFreq[i] = float64(countTermFreq(q, o.corpus[docID]))
        }

        idf, err := o.IDF(q)
        if err != nil {
            if o.logger != nil {
                o.logger.Printf("Error calculating IDF for term '%s': %v", q, err)
            }
            continue
        }

        for i, docID := range docIDs {
            if docID < 0 || docID >= o.corpusSize {
                continue
            }
            docLen := o.docLengths[docID]
            k := o.k1 * (1 - o.b + o.b*float64(docLen)/o.avgDocLen)
            scores[i] += idf * (qFreq[i] / (qFreq[i] + k))
        }
    }

    return scores, nil
}

// GetTopN returns the top N documents for the given query.
func (o *BM25Okapi) GetTopN(query []string, n int) ([]string, error) {
    if len(query) == 0 {
        return nil, errors.New("query cannot be empty")
    }

    if n <= 0 {
        if o.logger != nil {
            o.logger.Printf("Invalid value for n: %d. Returning empty slice.", n)
        }
        return []string{}, nil
    }

    scores, err := o.GetScores(query)
    if err != nil {
        return nil, err
    }

    topNIndices := topNIndices(scores, n)

    topDocs := make([]string, len(topNIndices))
    for i, idx := range topNIndices {
        topDocs[i] = joinTokens(o.corpus[idx])
    }

    return topDocs, nil
}
