package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/kitagry/go-nlp/utils"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type LDA struct {
	TopicNum int
	Alpha    float64
	Beta     float64

	nMZ *mat.Dense
	nZT *mat.Dense
	nZ  []float64
	zMN [][]int

	docIds    [][]float64
	id2Word   []string
	vocabSize int
}

func NewLDA(topicNum int, alpha, beta float64) LDA {
	return LDA{
		TopicNum: topicNum,
		Alpha:    alpha,
		Beta:     beta,
	}
}

func (lda *LDA) SetDocs(docIds [][]float64, id2Word []string) {
	lda.docIds = docIds
	lda.id2Word = id2Word
	vocabularySize := len(id2Word)
	M := len(docIds)
	lda.vocabSize = vocabularySize

	lda.zMN = make([][]int, len(docIds))
	lda.nMZ = mat.NewDense(M, lda.TopicNum, initializeArray(M*lda.TopicNum, lda.Alpha))
	lda.nZT = mat.NewDense(lda.TopicNum, lda.vocabSize, initializeArray(lda.TopicNum*lda.vocabSize, lda.Beta))
	lda.nZ = initializeArray(lda.TopicNum, float64(lda.vocabSize)*lda.Beta)

	rand.Seed(time.Now().UnixNano())
	for m, docId := range docIds {
		zN := make([]int, len(docId))
		for i := 0; i < len(docId); i++ {
			zN[i] = rand.Intn(lda.TopicNum)
		}
		lda.zMN[m] = zN

		for i, t := range docId {
			z := zN[i]
			lda.nMZ.Set(m, z, lda.nMZ.At(m, z)+1)
			lda.nZT.Set(z, int(t), lda.nZT.At(z, int(t))+1)
			lda.nZ[z] += 1
		}
	}
}

func (lda *LDA) Inference() {
	for m, docId := range lda.docIds {
		zN := lda.zMN[m]
		for n, t := range docId {
			z := zN[n]
			lda.nMZ.Set(m, z, lda.nMZ.At(m, z)-1)
			lda.nZT.Set(z, int(t), lda.nZT.At(z, int(t))-1)
			lda.nZ[z] -= 1

			// トピックの事後確率からサンプリングを行う
			vd := mat.NewVecDense(lda.TopicNum, nil)
			vd.MulElemVec(lda.nZT.ColView(int(t)), lda.nMZ.RowView(m))
			subVec := mat.NewVecDense(lda.TopicNum, lda.nZ)
			vd.SubVec(vd, subVec)
			vd.DivElemVec(vd, mat.NewVecDense(lda.TopicNum, initializeArray(lda.TopicNum, sum(vd.RawVector().Data))))
			c := distuv.NewCategorical(vd.RawVector().Data, nil)
			newZ := int(c.Rand())

			zN[n] = newZ
			lda.nMZ.Set(m, z, lda.nMZ.At(m, z)+1)
			lda.nZT.Set(newZ, int(t), lda.nZT.At(newZ, int(t))+1)
			lda.nZ[newZ] += 1
		}
	}
}

func (lda *LDA) Worddist() *mat.Dense {
	row, col := lda.nZT.Dims()
	ar := make([]float64, row*col)
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			ar[i*col+j] = lda.nZ[i]
		}
	}
	res := mat.NewDense(row, col, nil)
	res.DivElem(lda.nZT, mat.NewDense(row, col, ar))
	return res
}

func initializeArray(size int, scalar float64) []float64 {
	array := make([]float64, size)
	for i := 0; i < size; i++ {
		array[i] = scalar
	}
	return array
}

func sum(arr []float64) float64 {
	res := 0.
	for _, ar := range arr {
		res += ar
	}
	return res
}

func in(x float64, list []float64) bool {
	for _, el := range list {
		if x == el {
			return true
		}
	}
	return false
}

func outputWordTopicDist(lda LDA) {
	zcount := make([]int, lda.TopicNum)
	wordCount := make([]map[int]int, lda.TopicNum)
	for i := 0; i < lda.TopicNum; i++ {
		wordCount[i] = make(map[int]int)
	}

	for i := 0; i < len(lda.docIds); i++ {
		xlist := lda.docIds[i]
		zlist := lda.zMN[i]
		for j := 0; j < len(xlist); j++ {
			x := xlist[j]
			z := zlist[j]
			zcount[z] += 1
			if _, ok := wordCount[z][int(x)]; ok {
				wordCount[z][int(x)] += 1
			} else {
				wordCount[z][int(x)] = 1
			}
		}
	}

	phi := lda.Worddist()
	for k := 0; k < lda.TopicNum; k++ {
		fmt.Printf("\ntopic %d (%d words)\n", k, zcount[k])
		row := phi.RawRowView(k)
		row = reversePosiNega(row)
		args := argsort(row)
		if len(args) > 5 {
			args = args[:5]
		}
		for _, w := range args {
			word, _ := wordCount[k][w]
			fmt.Printf("%s: %f (%d)\n", lda.id2Word[w], phi.At(k, w), word)
		}
	}
}

func argsort(ar []float64) []int {
	type st struct {
		value float64
		arg   int
	}

	array := make([]st, len(ar))
	for i, el := range ar {
		array[i] = st{value: el, arg: i}
	}

	sort.SliceStable(array, func(i, j int) bool {
		return array[i].value < array[j].value
	})

	result := make([]int, len(ar))
	for i, el := range array {
		result[i] = el.arg
	}
	return result
}

func reversePosiNega(ar []float64) []float64 {
	result := make([]float64, len(ar))
	for i := 0; i < len(ar); i++ {
		result[i] = ar[i] * -1
	}
	return result
}

func main() {
	f, err := os.Open("./movie.txt")
	if err != nil {
		panic(err)
	}

	sc := bufio.NewScanner(f)
	titles := make([]string, 0)
	docs := make([]string, 0)
	for sc.Scan() {
		t := sc.Text()

		texts := strings.SplitN(t, ",", 2)
		titles = append(titles, texts[0])
		docs = append(docs, texts[1])
	}

	fmt.Println("Preprocess documents.")
	preprocessedDocs := make([][]string, len(docs))
	for i := 0; i < len(docs); i++ {
		preprocessedDocs[i] = utils.Preprocessing(docs[i])
	}

	word2Vectorizer := utils.NewWord2IdVectorizer()
	vecs := word2Vectorizer.Vectorize(preprocessedDocs)
	id2Word := word2Vectorizer.Id2Word()
	topicNum := 12
	lda := NewLDA(topicNum, 0.5, 0.5)
	lda.SetDocs(vecs, id2Word)

	for i := 0; i < 200; i++ {
		lda.Inference()
	}

	outputWordTopicDist(lda)

	groupedTitles := make([][]string, topicNum)
	for i := 0; i < topicNum; i++ {
		groupedTitles[i] = make([]string, 0)
	}

	for i, title := range titles {
		probabilities := lda.nMZ.RawRowView(i)
		groupedTitles[maxIndex(probabilities)] = append(groupedTitles[maxIndex(probabilities)], title)
	}

	for i, titles := range groupedTitles {
		fmt.Printf("\ntopic %d:\n", i)
		fmt.Print("[")
		for _, title := range titles {
			fmt.Printf(" %s,", title)
		}
		fmt.Print("]\n")
	}
}

func maxIndex(arr []float64) int {
	max := arr[0]
	maxIndex := 0
	for i, el := range arr {
		if el > max {
			max = el
			maxIndex = i
		}
	}
	return maxIndex
}
