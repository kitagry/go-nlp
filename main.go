package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"

	"github.com/aaaton/golem"
	"github.com/aaaton/golem/dicts/en"
)

func write(docs [][]string, countries []string, bowVec, tfidfVec [][]float64) error {
	f, err := os.OpenFile("result.csv", os.O_WRONLY|os.O_CREATE, 0600)
	if err != nil {
		return err
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
			return err
		}
		result[1] = fmt.Sprint(dist)

		dist, err = cosineDistance(bowVec[1], bowVec[i])
		if err != nil {
			return err
		}
		result[2] = fmt.Sprint(dist)

		dist, err = euclideanDistance(tfidfVec[1], tfidfVec[i])
		if err != nil {
			return err
		}
		result[3] = fmt.Sprint(dist)

		dist, err = cosineDistance(tfidfVec[1], tfidfVec[i])
		if err != nil {
			return err
		}
		result[4] = fmt.Sprint(dist)

		writer.Write(result)
	}
	writer.Flush()
	return nil
}

type offsetUnitGrid struct {
	XOffset, YOffset float64

	Data mat.Matrix
}

func (g offsetUnitGrid) Dims() (c, r int)   { r, c = g.Data.Dims(); return c, r }
func (g offsetUnitGrid) Z(c, r int) float64 { return g.Data.At(r, c) }
func (g offsetUnitGrid) X(c int) float64 {
	_, n := g.Data.Dims()
	if c < 0 || c >= n {
		panic("column index out of range")
	}
	return float64(c) + g.XOffset
}
func (g offsetUnitGrid) Y(r int) float64 {
	m, _ := g.Data.Dims()
	if r < 0 || r >= m {
		panic("row index out of range")
	}
	return float64(r) + g.YOffset
}

func max(s ...float64) float64 {
	maxContent := s[0]
	for _, el := range s {
		if el > maxContent {
			maxContent = el
		}
	}
	return maxContent
}

func min(s ...float64) float64 {
	minContent := s[0]
	for _, el := range s {
		if el < minContent {
			minContent = el
		}
	}
	return minContent
}

func saveImage(imageName string, labels []string, data []float64) {
	// 正規化
	minData := min(data...)
	maxData := max(data...)
	for i := 0; i < len(data); i++ {
		data[i] = (data[i] - minData) / maxData * 10
	}

	m := offsetUnitGrid{
		XOffset: 0,
		YOffset: 0,
		Data:    mat.NewDense(16, 16, data)}
	pal := palette.Heat(10, 1)
	h := plotter.NewHeatMap(m, pal)

	p, err := plot.New()
	if err != nil {
		log.Panic(err)
	}
	p.Title.Text = imageName

	p.NominalX(labels[1:]...)
	p.NominalY(labels[1:]...)

	p.Add(h)

	// Create a legend.
	l, err := plot.NewLegend()
	if err != nil {
		log.Panic(err)
	}
	thumbs := plotter.PaletteThumbnailers(pal)
	for i := len(thumbs) - 1; i >= 0; i-- {
		t := thumbs[i]
		if i != 0 && i != len(thumbs)-1 {
			l.Add("", t)
			continue
		}
		var val float64
		switch i {
		case 0:
			val = h.Min
		case len(thumbs) - 1:
			val = h.Max
		}
		l.Add(fmt.Sprintf("%.2g", val), t)
	}

	// Change the rotation of the X tick labels to make them fit better.
	p.X.Tick.Label.Rotation = math.Pi / 5
	p.X.Tick.Label.YAlign = draw.YCenter
	p.X.Tick.Label.XAlign = draw.XRight

	img := vgimg.New(550, 500)
	dc := draw.New(img)

	l.Top = true
	// Calculate the width of the legend.
	r := l.Rectangle(dc)
	legendWidth := r.Max.X - r.Min.X
	l.YOffs = -p.Title.Font.Extents().Height // Adjust the legend down a little.

	l.Draw(dc)
	dc = draw.Crop(dc, 0, -legendWidth-vg.Millimeter, 0, 0) // Make space for the legend.
	p.Draw(dc)
	w, err := os.Create(imageName + ".png")
	if err != nil {
		log.Panic(err)
	}
	png := vgimg.PngCanvas{Canvas: img}
	if _, err = png.WriteTo(w); err != nil {
		log.Panic(err)
	}
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

	bowEuclidData := make([]float64, 16*16)
	bowCosineData := make([]float64, 16*16)
	tfidfEuclidData := make([]float64, 16*16)
	tfidfCosineData := make([]float64, 16*16)
	for i := 1; i < len(docs); i++ {
		for j := 1; j < len(docs); j++ {
			d, err := euclideanDistance(bowVec[i], bowVec[j])
			if err != nil {
				fmt.Println(err)
				return
			}
			bowEuclidData[(i-1)*16+j-1] = d

			d, err = cosineDistance(bowVec[i], bowVec[j])
			if err != nil {
				fmt.Println(err)
				return
			}
			bowCosineData[(i-1)*16+j-1] = d

			d, err = euclideanDistance(tfidfVec[i], tfidfVec[j])
			if err != nil {
				fmt.Println(err)
				return
			}
			tfidfEuclidData[(i-1)*16+j-1] = d

			d, err = cosineDistance(tfidfVec[i], tfidfVec[j])
			if err != nil {
				fmt.Println(err)
				return
			}
			tfidfCosineData[(i-1)*16+j-1] = d
		}
	}
	saveImage("bow-euclid", countries, bowEuclidData)
	saveImage("bow-cosine", countries, bowCosineData)
	saveImage("tfidf-euclid", countries, tfidfEuclidData)
	saveImage("tfidf-cosine", countries, tfidfCosineData)
}
