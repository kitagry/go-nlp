package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"

	"github.com/aaaton/golem"
	"github.com/aaaton/golem/dicts/en"
	"github.com/mdanzinger/stopwords"
)

func cleaningText(text string) string {
	re := regexp.MustCompile("(@|<.+?>|\\(.*\\))")
	return re.ReplaceAllString(text, "")
}

func removeStopWords(text string) string {
	return stopwords.CleanString(text, "en", false)
}

func tokenizeText(text string) []string {
	return strings.Split(text, " ")
}

var lemmatizer *golem.Lemmatizer

func lemmatizeWord(word string) string {
	return lemmatizer.Lemma(word)
}

func preprocessing(text string) []string {
	text = cleaningText(text)
	text = removeStopWords(text)
	if text[0] == ' ' {
		text = text[1:]
	}
	if text[len(text)-1] == ' ' {
		text = text[0 : len(text)-1]
	}
	texts := tokenizeText(text)
	for i := 0; i < len(texts); i++ {
		texts[i] = lemmatizeWord(texts[i])
	}
	return texts
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

func bowVectorizer(docs [][]string) [][]float64 {
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

func tfidfVectorizer(docs [][]string) [][]float64 {
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

func euclideanDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("配列の長さは同じにしてください")
	}

	count := 0.
	for i := 0; i < len(a); i++ {
		count += math.Pow(a[i]-b[i], 2)
	}
	return math.Sqrt(count), nil
}

func inner(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("配列の長さは同じにしてください")
	}
	result := 0.
	for i := 0; i < len(a); i++ {
		result += a[i] * b[i]
	}
	return result, nil
}

func norm(a []float64) float64 {
	result := 0.
	for _, p := range a {
		result += math.Pow(p, 2)
	}
	return math.Sqrt(result)
}

func cosineDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("配列の長さは同じにしてください")
	}
	inner, _ := inner(a, b)
	normA := norm(a)
	normB := norm(b)
	return inner / (normA * normB), nil
}

func main() {
	var err error
	lemmatizer, err = golem.New(en.New())
	if err != nil {
		panic(err)
	}

	f, err := os.Open("nlp_country.csv")
	if err != nil {
		fmt.Println("ファイル読み込みエラー")
		return
	}
	defer f.Close()
	reader := csv.NewReader(f)

	countries := make([]string, 0)
	documents := make([]string, 0)
	for {
		line, err := reader.Read()
		if err != nil {
			break
		}
		countries = append(countries, line[0])
		documents = append(documents, line[1])
	}

	docs := make([][]string, len(documents))
	for index, doc := range documents {
		docs[index] = preprocessing(doc)
	}

	bowVec := bowVectorizer(docs)
	tfidfVec := tfidfVectorizer(docs)

	f, err = os.OpenFile("result.csv", os.O_WRONLY|os.O_CREATE, 0600)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer f.Close()
	writer := csv.NewWriter(f)

	header := []string{"Japanとの類似度", "BoW ユークリッド距離", "BoW コサイン距離", "tfidf ユークリッド距離", "tfidf コサイン距離"}
	writer.Write(header)

	for i := 2; i < len(docs); i++ {
		result := make([]string, 5)
		result[0] = countries[i]

		dist, err := euclideanDistance(bowVec[1], bowVec[i])
		if err != nil {
			fmt.Println(err)
			return
		}
		result[1] = fmt.Sprint(dist)

		dist, err = cosineDistance(bowVec[1], bowVec[i])
		if err != nil {
			fmt.Println(err)
			return
		}
		result[2] = fmt.Sprint(dist)

		dist, err = euclideanDistance(tfidfVec[1], tfidfVec[i])
		if err != nil {
			fmt.Println(err)
			return
		}
		result[3] = fmt.Sprint(dist)

		dist, err = cosineDistance(tfidfVec[1], tfidfVec[i])
		if err != nil {
			fmt.Println(err)
			return
		}
		result[4] = fmt.Sprint(dist)

		writer.Write(result)
	}
	writer.Flush()
}
