package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"math"
	"os"

	"github.com/aaaton/golem"
	"github.com/aaaton/golem/dicts/en"
)

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
		docs[index] = Preprocessing(doc)
	}

	bowVectorizer := NewBowVectorizer()
	bowVec := bowVectorizer.Vectorize(docs)
	tfidfVectorizer := NewTfidfVectorizer()
	tfidfVec := tfidfVectorizer.Vectorize(docs)

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
