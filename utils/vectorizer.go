package utils

import "math"

type Vectorizer interface {
	Vectorize(docs [][]string) [][]float64
}

func makeWord2Id(docs [][]string) map[string]int {
	word2id := make(map[string]int)
	count := 0
	for _, doc := range docs {
		for _, word := range doc {
			if _, ok := word2id[word]; !ok {
				word2id[word] = count
				count++
			}
		}
	}
	return word2id
}

type bowVectorizer struct{}

func NewBowVectorizer() Vectorizer {
	return bowVectorizer{}
}

func (bv bowVectorizer) Vectorize(docs [][]string) [][]float64 {
	word2id := makeWord2Id(docs)

	results := make([][]float64, len(docs))
	for index, doc := range docs {
		result := make([]float64, len(word2id))
		for _, word := range doc {
			id, ok := word2id[word]
			if !ok {
				panic("なんかおかしい")
			}
			result[id] += 1
		}
		results[index] = result
	}
	return results
}

type tfidfVectorizer struct{}

func NewTfidfVectorizer() Vectorizer {
	return tfidfVectorizer{}
}

func countWords(doc []string) map[string]int {
	maps := make(map[string]int)
	for _, word := range doc {
		if _, ok := maps[word]; ok {
			maps[word] += 1
		} else {
			maps[word] = 1
		}
	}
	return maps
}

func in(word string, doc []string) bool {
	for _, t := range doc {
		if t == word {
			return true
		}
	}
	return false
}

func count(word string, docs [][]string) (count int) {
	for _, doc := range docs {
		if in(word, doc) {
			count++
		}
	}
	return
}

func (tv tfidfVectorizer) Vectorize(docs [][]string) [][]float64 {
	tf := func(word2id map[string]int, doc []string) []float64 {
		termCounts := make([]float64, len(word2id))
		for key, count := range countWords(doc) {
			id, ok := word2id[key]
			if !ok {
				panic("おかしいよ")
			}
			termCounts[id] = float64(count) / float64(len(doc))
		}
		return termCounts
	}

	idf := func(word2id map[string]int, docs [][]string) []float64 {
		idf := make([]float64, len(word2id))
		for word, id := range word2id {
			idf[id] = math.Log(float64(len(docs)) / float64(count(word, docs)))
		}
		return idf
	}

	word2id := makeWord2Id(docs)

	tfidfVec := make([][]float64, len(docs))
	idfVec := idf(word2id, docs)
	for i := 0; i < len(docs); i++ {
		res := tf(word2id, docs[i])
		for j := 0; j < len(res); j++ {
			res[j] *= idfVec[j]
		}
		tfidfVec[i] = res
	}
	return tfidfVec
}
