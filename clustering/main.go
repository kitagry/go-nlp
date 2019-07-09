package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/kitagry/go-nlp/utils"
)

func kMeans(vecs [][]float64, knum, nSample int) (result []int) {
	result = make([]int, len(vecs))

	// ランダムにグループ分け
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < len(result); i++ {
		result[i] = rand.Intn(knum)
	}

	for sample := 0; sample < nSample; sample++ {
		// 重心を求める
		centers := make([][]float64, knum)
		for i := 0; i < knum; i++ {
			centers[i] = make([]float64, len(vecs[0]))
		}

		for index, num := range result {
			for i := 0; i < len(vecs[0]); i++ {
				centers[num][i] += vecs[index][i]
			}
		}

		for i := 0; i < len(vecs); i++ {
			vec := vecs[i]
			minIndex := 0
			minDistance := -1.
			for i, center := range centers {
				dis, err := utils.CosineDistance(vec, center)
				if err != nil {
					fmt.Println(err)
					return
				}

				if minDistance < dis {
					minIndex = i
					minDistance = dis
				}
			}
			result[i] = minIndex
		}
	}

	return
}

func main() {
	f, err := os.Open("./enwiki-20150112-400-r10-105752.txt")
	if err != nil {
		panic(err)
	}
	DOCNUM := 500

	sc := bufio.NewScanner(f)
	titles := make([]string, DOCNUM)
	docs := make([]string, DOCNUM)
	isTitle := true
	i := 0
	for sc.Scan() {
		t := sc.Text()

		if isTitle {
			if t == "" {
				isTitle = false
				continue
			}
			titles[i] = t
		} else {
			if t == "" {
				sc.Scan()
				sc.Text()
				isTitle = true
				i++
				if i == DOCNUM {
					break
				}
				continue
			}
			docs[i] += t
		}
	}

	preprocessedDocs := make([][]string, len(docs))
	for i := 0; i < len(docs); i++ {
		preprocessedDocs[i] = utils.Preprocessing(docs[i])
	}

	tfidfVectorizer := utils.NewTfidfVectorizer(0.1)
	vecs := tfidfVectorizer.Vectorize(preprocessedDocs)

	fmt.Println(kMeans(vecs, 10, 10))
}
