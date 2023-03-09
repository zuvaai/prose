// Copyright 2013 Matthew Honnibal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package prose

import (
	"fmt"
	"math"
	"path"
	"regexp"
	"strconv"
	"strings"
)

// TupleSlice is a slice of tuples in the form (words, tags).
type TupleSlice [][][]string

// Len returns the length of a Tuple.
func (t TupleSlice) Len() int { return len(t) }

// Swap switches the ith and jth elements in a Tuple.
func (t TupleSlice) Swap(i, j int) { t[i], t[j] = t[j], t[i] }

// ReadTagged converts pre-tagged input into a TupleSlice suitable for training.
func ReadTagged(text, sep string) TupleSlice {
	lines := strings.Split(text, "\n")
	length := len(lines)
	t := make(TupleSlice, length)
	for i, sent := range lines {
		set := strings.Split(sent, " ")
		length = len(set)
		tokens := make([]string, length)
		tags := make([]string, length)
		for j, token := range set {
			parts := strings.Split(token, sep)
			tokens[j] = parts[0]
			tags[j] = parts[1]
		}
		t[i] = [][]string{tokens, tags}
	}
	return t
}

var none = regexp.MustCompile(`^(?:0|\*[\w?]\*|\*\-\d{1,3}|\*[A-Z]+\*\-\d{1,3}|\*)$`)
var keep = regexp.MustCompile(`^\-[A-Z]{3}\-$`)

// averagedPerceptron is a Averaged Perceptron classifier.
type averagedPerceptron struct {
	classes  []string
	classMap map[string]int
	stamps   map[string]float64
	totals   map[string]float64
	tagMap   map[string]string
	//	weights       map[string]map[string]float64
	linearWeights map[string][]float64

	// TODO: Training
	//
	// instances float64
}

// newAveragedPerceptron creates a new AveragedPerceptron model.
func newAveragedPerceptron(tags map[string]string, classes []string, linearWeights map[string][]float64) *averagedPerceptron {
	cm := make(map[string]int, len(classes))
	for i := range classes {
		cm[classes[i]] = i
	}
	return &averagedPerceptron{
		totals: make(map[string]float64), stamps: make(map[string]float64),
		classes: classes, tagMap: tags, classMap: cm, linearWeights: linearWeights}
}

/* TODO: Training API

"github.com/shogo82148/go-shuffle"

// marshal saves the model to disk.
func (m *averagedPerceptron) marshal(path string) error {
	folder := filepath.Join(path, "AveragedPerceptron")
	err := os.Mkdir(folder, os.ModePerm)
	for i, entry := range []string{"weights", "tags", "classes"} {
		component, _ := os.Create(filepath.Join(folder, entry+".gob"))
		encoder := gob.NewEncoder(component)
		if i == 0 {
			checkError(encoder.Encode(m.weights))
		} else if i == 1 {
			checkError(encoder.Encode(m.tagMap))
		} else {
			checkError(encoder.Encode(m.classes))
		}
	}
	return err
}

// train an Averaged Perceptron model based on sentences.
func (pt *perceptronTagger) train(sentences TupleSlice, iterations int) {
	var guess string
	var found bool

	pt.makeTagMap(sentences)
	for i := 0; i < iterations; i++ {
		for _, tuple := range sentences {
			words, tags := tuple[0], tuple[1]
			p1, p2 := "-START-", "-START2-"
			context := []string{p1, p2}
			for _, w := range words {
				if w == "" {
					continue
				}
				context = append(context, normalize(w))
			}
			context = append(context, []string{"-END-", "-END2-"}...)
			for i, word := range words {
				if guess, found = pt.tagMap[word]; !found {
					feats := featurize(i, context, word, p1, p2)
					guess = pt.model.predict(feats)
					pt.model.update(tags[i], guess, feats)
				}
				p2 = p1
				p1 = guess
			}
		}
		shuffle.Shuffle(sentences)
	}
	pt.model.averageWeights()
}

func (m *averagedPerceptron) averageWeights() {
	for feat, weights := range m.weights {
		newWeights := make(map[string]float64)
		for class, weight := range weights {
			key := feat + "-" + class
			total := m.totals[key]
			total += (m.instances - m.stamps[key]) * weight
			averaged, _ := stats.Round(total/m.instances, 3)
			if averaged != 0.0 {
				newWeights[class] = averaged
			}
		}
		m.weights[feat] = newWeights
	}
}

// newTrainedPerceptronTagger creates a new PerceptronTagger using the given
// model.
func newTrainedPerceptronTagger(model *averagedPerceptron) *perceptronTagger {
	return &perceptronTagger{model: model}
}

func (pt *perceptronTagger) makeTagMap(sentences TupleSlice) {
	counts := make(map[string]map[string]int)
	for _, tuple := range sentences {
		words, tags := tuple[0], tuple[1]
		for i, word := range words {
			tag := tags[i]
			if counts[word] == nil {
				counts[word] = make(map[string]int)
			}
			counts[word][tag]++
			pt.model.addClass(tag)
		}
	}
	for word, tagFreqs := range counts {
		tag, mode := maxValue(tagFreqs)
		n := float64(sumValues(tagFreqs))
		if n >= 20 && (float64(mode)/n) >= 0.97 {
			pt.tagMap[word] = tag
		}
	}
}

func sumValues(m map[string]int) int {
	sum := 0
	for _, v := range m {
		sum += v
	}
	return sum
}

func maxValue(m map[string]int) (string, int) {
	maxValue := 0
	key := ""
	for k, v := range m {
		if v >= maxValue {
			maxValue = v
			key = k
		}
	}
	return key, maxValue
}

func get(k string, m map[string]float64) float64 {
	if v, ok := m[k]; ok {
		return v
	}
	return 0.0
}

func (m *averagedPerceptron) update(truth, guess string, feats map[string]float64) {
	m.instances++
	if truth == guess {
		return
	}
	for f := range feats {
		weights := make(map[string]float64)
		if val, ok := m.weights[f]; ok {
			weights = val
		} else {
			m.weights[f] = weights
		}
		m.updateFeat(truth, f, get(truth, weights), 1.0)
		m.updateFeat(guess, f, get(guess, weights), -1.0)
	}
}

func (m *averagedPerceptron) updateFeat(c, f string, v, w float64) {
	key := f + "-" + c
	m.totals[key] = (m.instances - m.stamps[key]) * w
	m.stamps[key] = m.instances
	m.weights[f][c] = w + v
}

func (m *averagedPerceptron) addClass(class string) {
	if !stringInSlice(class, m.classes) {
		m.classes = append(m.classes, class)
	}
}*/

// perceptronTagger is a port of Textblob's "fast and accurate" POS tagger.
// See https://github.com/sloria/textblob-aptagger for details.
type PerceptronTagger struct {
	model *averagedPerceptron
}

// newPerceptronTagger creates a new PerceptronTagger and loads the built-in
// AveragedPerceptron model.
func NewPerceptronTagger() (*PerceptronTagger, error) {
	var tags map[string]string
	var classes []string
	var lwts map[string][]float64

	dec, err := ReadAndDecodeBytes(path.Join("AveragedPerceptron", "classes.gob"))
	if err != nil {
		return nil, fmt.Errorf("unable to read classes: %w", err)
	}
	err = dec.Decode(&classes)
	if err != nil {
		return nil, fmt.Errorf("unable to decode classes: %w", err)
	}

	dec, err = ReadAndDecodeBytes(path.Join("AveragedPerceptron", "tags.gob"))
	if err != nil {
		return nil, fmt.Errorf("unable to read tags: %w", err)
	}
	err = dec.Decode(&tags)
	if err != nil {
		return nil, fmt.Errorf("unable to decode tags: %w", err)
	}

	dec, err = ReadAndDecodeBytes(path.Join("AveragedPerceptron", "weights-linear.gob"))
	if err != nil {
		return nil, fmt.Errorf("unable to read linear weights: %w", err)
	}
	err = dec.Decode(&lwts)
	if err != nil {
		return nil, fmt.Errorf("unable to decode lienar weights: %w", err)
	}

	return &PerceptronTagger{model: newAveragedPerceptron(tags, classes, lwts)}, nil
}

// Tag takes a slice of words and returns a slice of tagged tokens.
func (pt *PerceptronTagger) Tag(tokens []*Token) []*Token {
	var tag string
	var found bool

	p1, p2 := "-START-", "-START2-"
	length := len(tokens) + 4
	context := make([]string, length)
	context[0] = p1
	context[1] = p2
	for i, t := range tokens {
		context[i+2] = normalize(t.Text)
	}
	context[length-2] = "-END-"
	context[length-1] = "-END2-"
	for i := 0; i < len(tokens); i++ {
		word := tokens[i].Text
		if word == "-" {
			tag = "-"
		} else if _, ok := emoticons[word]; ok {
			tag = "SYM"
		} else if strings.HasPrefix(word, "@") {
			// TODO: URLs and emails?
			tag = "NN"
		} else if none.MatchString(word) {
			tag = "-NONE-"
		} else if keep.MatchString(word) {
			tag = word
		} else if tag, found = pt.model.tagMap[word]; !found {
			tag = pt.model.predict(featurize(i, context, word, p1, p2))
		}
		tokens[i].Tag = tag
		p2 = p1
		p1 = tag
	}

	return tokens
}

func (m *averagedPerceptron) predict(features [14]string) string {
	var weights []float64
	var found bool

	scores := make([]float64, len(m.classes))
	for _, feat := range features {
		// if we re-add training then also check feat.cnt == 0 {
		if weights, found = m.linearWeights[feat]; !found {
			continue
		}
		for label, weight := range weights {
			// If we add training then scores[label] += feat.cnt * weight
			scores[label] += weight
		}
	}
	return m.classes[max(scores)]
}

func max(scores []float64) int {
	var class int
	max := math.Inf(-1)
	for label, value := range scores {
		if value > max {
			max = value
			class = label
		}
	}
	return class
}

// We may need this struct if we want to add back training code
// type freq struct {
// 	feat string
// 	cnt  float64
// }

func featurize(i int, ctx []string, w, p1, p2 string) [14]string {
	feats := [14]string{}
	suf := min(len(w), 3)
	i = min(len(ctx)-2, i+2)
	iminus := min(len(ctx[i-1]), 3)
	iplus := min(len(ctx[i+1]), 3)
	feats[0] = "bias"
	feats[1] = strings.Join([]string{"i suffix", w[len(w)-suf:]}, " ")
	feats[2] = strings.Join([]string{"i pref1", string(w[0])}, " ")
	feats[3] = strings.Join([]string{"i-1 tag", p1}, " ")
	feats[4] = strings.Join([]string{"i-2 tag", p2}, " ")
	feats[5] = strings.Join([]string{"i tag+i-2 tag", p1, p2}, " ")
	feats[6] = strings.Join([]string{"i word", ctx[i]}, " ")
	feats[7] = strings.Join([]string{"i-1 tag+i word", p1, ctx[i]}, " ")
	feats[8] = strings.Join([]string{"i-1 word", ctx[i-1]}, " ")
	feats[9] = strings.Join([]string{"i-1 suffix", ctx[i-1][len(ctx[i-1])-iminus:]}, " ")
	feats[10] = strings.Join([]string{"i-2 word", ctx[i-2]}, " ")
	feats[11] = strings.Join([]string{"i+1 word", ctx[i+1]}, " ")
	feats[12] = strings.Join([]string{"i+1 suffix", ctx[i+1][len(ctx[i+1])-iplus:]}, " ")
	feats[13] = strings.Join([]string{"i+2 word", ctx[i+2]}, " ")
	return feats
}

func normalize(word string) string {
	if word == "" {
		return word
	}
	first := string(word[0])
	if strings.Contains(word, "-") && first != "-" {
		return "!HYPHEN"
	} else if _, err := strconv.Atoi(word); err == nil && len(word) == 4 {
		return "!YEAR"
	} else if _, err := strconv.Atoi(first); err == nil {
		return "!DIGITS"
	}
	return strings.ToLower(word)
}
