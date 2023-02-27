package prose

import (
	"encoding/gob"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"gonum.org/v1/gonum/mat"
)

var maxLogDiff = math.Log2(1e-30)

type mappedProbDist struct {
	dict map[string]*probEnc
	log  bool
}

func (m *mappedProbDist) prob(label string) float64 {
	if p, found := m.dict[label]; found {
		return math.Pow(2, p.prob)
	}
	return 0.0
}

func newMappedProbDist(dict map[string]*probEnc, normalize bool) *mappedProbDist {
	if normalize {
		values := make([]float64, len(dict))
		i := 0
		for _, v := range dict {
			values[i] = v.prob
			i++
		}
		sum := sumLogs(values)
		if sum <= math.Inf(-1) {
			p := math.Log2(1.0 / float64(len(dict)))
			for _, pe := range dict {
				pe.prob = p
			}
		} else {
			for _, pe := range dict {
				pe.prob -= sum
			}
		}
	}
	return &mappedProbDist{dict: dict, log: true}
}

type encodedValue struct {
	key   int
	value int
}

type feature struct {
	label    string
	features [17]string
}

type featureSet []feature

var featureOrder = []string{
	"bias", "en-wordlist", "nextpos", "nextword", "pos", "pos+prevtag",
	"prefix3", "prevpos", "prevtag", "prevword", "shape", "shape+prevtag",
	"suffix3", "word", "word+nextpos", "word.lower", "wordlen"}

// binaryMaxentClassifier is a feature encoding that generates vectors
// containing binary joint-features of the form:
//
//	|  joint_feat(fs, l) = { 1 if (fs[fname] == fval) and (l == label)
//	|                      {
//	|                      { 0 otherwise
//
// where `fname` is the name of an input-feature, `fval` is a value for that
// input-feature, and `label` is a label.
//
// See https://www.nltk.org/_modules/nltk/classify/maxent.html for more
// information.
type binaryMaxentClassifier struct {
	cardinality int
	labels      []string
	mapping     map[string]int
	weights     []float64
	buf         []byte
}

// newMaxentClassifier creates a new binaryMaxentClassifier from the provided
// input values.
func newMaxentClassifier(
	weights []float64,
	mapping map[string]int,
	labels []string) *binaryMaxentClassifier {

	set := make(map[string]struct{})
	for label := range mapping {
		k := strings.Split(label, "-")[0]
		set[k] = struct{}{}
	}

	return &binaryMaxentClassifier{
		len(set) + 1,
		labels,
		mapping,
		weights,
		[]byte{}}
}

// marshal saves the model to disk.
func (m *binaryMaxentClassifier) marshal(path string) error {
	folder := filepath.Join(path, "Maxent")
	err := os.Mkdir(folder, os.ModePerm)
	for i, entry := range []string{"labels", "mapping", "weights"} {
		component, _ := os.Create(filepath.Join(folder, entry+".gob"))
		encoder := gob.NewEncoder(component)
		if i == 0 {
			checkError(encoder.Encode(m.labels))
		} else if i == 1 {
			checkError(encoder.Encode(m.mapping))
		} else {
			checkError(encoder.Encode(m.weights))
		}
	}
	return err
}

// entityExtracter is a maximum entropy classifier.
//
// See https://www.nltk.org/_modules/nltk/classify/maxent.html for more
// information.
type entityExtracter struct {
	model *binaryMaxentClassifier
}

// newEntityExtracter creates a new entityExtracter using the default model.
func newEntityExtracter() *entityExtracter {
	var mapping map[string]int
	var weights []float64
	var labels []string

	dec := getAsset("Maxent", "mapping.gob")
	checkError(dec.Decode(&mapping))

	dec = getAsset("Maxent", "weights.gob")
	checkError(dec.Decode(&weights))

	dec = getAsset("Maxent", "labels.gob")
	checkError(dec.Decode(&labels))

	return &entityExtracter{model: newMaxentClassifier(weights, mapping, labels)}
}

// newTrainedEntityExtracter creates a new EntityExtracter using the given
// model.
func newTrainedEntityExtracter(model *binaryMaxentClassifier) *entityExtracter {
	return &entityExtracter{model: model}
}

// chunk finds named-entity "chunks" from the given, pre-labeled tokens.
func (e *entityExtracter) chunk(tokens []*Token) []Entity {
	entities := []Entity{}
	end := ""

	parts := []*Token{}
	idx := 0

	for _, tok := range tokens {
		label := tok.Label
		if (label != "O" && label != end) ||
			(idx > 0 && tok.Tag == parts[idx-1].Tag) ||
			(idx > 0 && tok.Tag == "CD" && parts[idx-1].Label != "O") {
			end = strings.Replace(label, "B", "I", 1)
			parts = append(parts, tok)
			idx++
		} else if (label == "O" && end != "") || label == end {
			// We've found the end of an entity.
			if label != "O" {
				parts = append(parts, tok)
			}
			entities = append(entities, coalesce(parts))

			end = ""
			parts = []*Token{}
			idx = 0
		}
	}

	return entities
}

func (m *binaryMaxentClassifier) byteJoin(a, b, c string, rjc rune) string {
	jc := byte(rjc)
	n := len(a) + len(b) + len(c) + 2
	if len(m.buf) < n {
		m.buf = make([]byte, n)
	}
	copy(m.buf[0:], []byte(a))
	m.buf[len(a)] = jc
	copy(m.buf[len(a)+1:], []byte(b))
	m.buf[len(a)+len(b)+1] = jc
	copy(m.buf[len(a)+len(b)+2:], []byte(c))
	res := string(m.buf[:n])
	return res
}

func (m *binaryMaxentClassifier) encode(features [17]string, label string) []encodedValue {
	encoding := make([]encodedValue, 0, 18)
	for i, key := range featureOrder {
		val := features[i]
		entry := m.byteJoin(key, val, label, '-')
		if ret, found := m.mapping[entry]; found {
			encoding = append(encoding, encodedValue{
				key:   ret,
				value: 1})
		}
	}
	return encoding
}

func (m *binaryMaxentClassifier) encodeGIS(features [17]string, label string) []encodedValue {
	encoding := m.encode(features, label)
	length := len(m.mapping)

	total := 0
	for _, v := range encoding {
		total += v.value
	}
	encoding = append(encoding, encodedValue{
		key:   length,
		value: m.cardinality - total})

	return encoding
}

func adjustPos(text string, start, end int) (int, int) {
	index, left, right := -1, 0, 0
	_ = strings.Map(func(r rune) rune {
		index++
		if unicode.IsSpace(r) {
			if index < start {
				left++
			}
			if index < end {
				right++
			}
			return -1
		}
		return r
	}, text)
	return start - left, end - right
}

func extractFeatures(tokens []*Token, history []string) []feature {
	features := make([]feature, len(tokens))
	for i := range tokens {
		features[i] = feature{
			label:    history[i],
			features: extract(i, tokens, history)}
	}
	return features
}

func assignLabels(tokens []*Token, entity *EntityContext) []string {
	history := make([]string, len(tokens))
	for i := range tokens {
		history[i] = "O"
	}

	if entity.Accept {
		for _, span := range entity.Spans {
			start, end := adjustPos(entity.Text, span.Start, span.End)
			index := 0
			for i, tok := range tokens {
				if index == start {
					history[i] = "B-" + span.Label
				} else if index > start && index < end {
					history[i] = "I-" + span.Label
				}
				index += len(tok.Text)
			}
		}
	}

	return history
}

func makeCorpus(data []EntityContext, tagger *PerceptronTagger, tokenizer Tokenizer) featureSet {
	corpus := featureSet{}
	for i := range data {
		entry := &data[i]
		tokens := tagger.Tag(tokenizer.Tokenize(entry.Text))
		history := assignLabels(tokens, entry)
		for _, element := range extractFeatures(tokens, history) {
			corpus = append(corpus, element)
		}
	}
	return corpus
}

func extracterFromData(corpus featureSet) *entityExtracter {
	encoding := encode(corpus)
	cInv := 1.0 / float64(encoding.cardinality)

	empfreq := empiricalCount(corpus, encoding)
	rows, _ := empfreq.Dims()

	unattested := []int{}
	for index := 0; index < rows; index++ {
		if empfreq.At(index, 0) == 0.0 {
			unattested = append(unattested, index)
		}
		empfreq.SetVec(index, math.Log2(empfreq.At(index, 0)))
	}

	weights := make([]float64, rows)
	for _, idx := range unattested {
		weights[idx] = math.Inf(-1)
	}
	encoding.weights = weights

	classifier := newTrainedEntityExtracter(encoding)
	for index := 0; index < 100; index++ {
		est := estCount(classifier, corpus, encoding)
		for _, idx := range unattested {
			est.SetVec(idx, est.AtVec(idx)+1)
		}
		rows, _ := est.Dims()
		for index := 0; index < rows; index++ {
			est.SetVec(index, math.Log2(est.At(index, 0)))
		}
		weights = classifier.model.weights

		est.SubVec(empfreq, est)
		est.ScaleVec(cInv, est)

		for index := 0; index < len(weights); index++ {
			weights[index] += est.AtVec(index)
		}

		classifier.model.weights = weights
	}

	return classifier
}

func estCount(
	classifier *entityExtracter,
	corpus featureSet,
	encoder *binaryMaxentClassifier,
) *mat.VecDense {
	count := mat.NewVecDense(len(encoder.mapping)+1, nil)
	for _, entry := range corpus {
		pdist := classifier.probClassify(entry.features)
		for _, pe := range pdist.dict {
			prob := math.Pow(2, pe.prob)
			for _, enc := range pe.vec {
				out := count.AtVec(enc.key) + (prob * float64(enc.value))
				count.SetVec(enc.key, out)
			}
		}
	}
	return count
}

func (e *entityExtracter) classify(tokens []*Token) []*Token {
	length := len(tokens)
	history := make([]string, 0, length)
	for i := 0; i < length; i++ {
		scores := make(map[string]float64)
		features := extract(i, tokens, history)
		for _, label := range e.model.labels {
			total := 0.0
			for _, encoded := range e.model.encode(features, label) {
				total += e.model.weights[encoded.key] * float64(encoded.value)
			}
			scores[label] = total
		}
		label := maxMap(scores)
		tokens[i].Label = label
		history = append(history, simplePOS(label))
	}
	return tokens
}

func maxMap(scores map[string]float64) string {
	var class string
	max := math.Inf(-1)
	for label, value := range scores {
		if value > max {
			max = value
			class = label
		}
	}
	return class
}

type probEnc struct {
	prob float64
	vec  []encodedValue
}

func (e *entityExtracter) probClassify(features [17]string) *mappedProbDist {
	scores := make(map[string]*probEnc, len(e.model.labels))
	for _, label := range e.model.labels {
		vec := e.model.encodeGIS(features, label)
		total := 0.0
		for _, entry := range vec {
			total += e.model.weights[entry.key] * float64(entry.value)
		}
		scores[label] = &probEnc{prob: total, vec: vec}
	}

	//&mappedProbDist{dict: scores, log: true}
	return newMappedProbDist(scores, true)
}

func parseEntities(ents []string) string {
	if stringInSlice("B-PERSON", ents) && len(ents) == 2 {
		// PERSON takes precedence because it's hard to identify.
		return "PERSON"
	}
	return strings.Split(ents[0], "-")[1]
}

func coalesce(parts []*Token) Entity {
	length := len(parts)
	labels := make([]string, length)
	tokens := make([]string, length)
	for i, tok := range parts {
		tokens[i] = tok.Text
		labels[i] = tok.Label
	}
	return Entity{
		Label: parseEntities(labels),
		Text:  strings.Join(tokens, " "),
	}
}

const NoneFeat = "None"

func extract(i int, ctx []*Token, history []string) [17]string {
	//feats := make(map[string]string)
	feats := [17]string{}
	word := ctx[i].Text
	prevShape := NoneFeat

	feats[0] = "True"
	feats[13] = word
	feats[4] = ctx[i].Tag
	feats[2] = isBasic(word)
	feats[15] = strings.ToLower(word)
	feats[12] = nSuffix(word, 3)
	feats[6] = nPrefix(word, 3)
	feats[10] = shape(word)
	feats[16] = strconv.Itoa(len(word))

	if i == 0 {
		feats[8] = NoneFeat
		feats[9], feats[7] = NoneFeat, NoneFeat
	} else if i == 1 {
		feats[9] = strings.ToLower(ctx[i-1].Text)
		feats[7] = ctx[i-1].Tag
		feats[8] = history[i-1]
	} else {
		feats[9] = strings.ToLower(ctx[i-1].Text)
		feats[7] = ctx[i-1].Tag
		feats[8] = history[i-1]
		prevShape = shape(ctx[i-1].Text)
	}

	if i == len(ctx)-1 {
		feats[3], feats[2] = NoneFeat, NoneFeat
	} else {
		feats[3] = strings.ToLower(ctx[i+1].Text)
		feats[2] = strings.ToLower(ctx[i+1].Tag)
	}

	feats[14] = strings.Join(
		[]string{feats[15], feats[2]}, "+")
	feats[5] = strings.Join(
		[]string{feats[4], feats[8]}, "+")
	feats[11] = strings.Join(
		[]string{prevShape, feats[8]}, "+")

	return feats
}

func shape(word string) string {
	if isNumeric(word) {
		return "number"
	} else if match, _ := regexp.MatchString(`\W+$`, word); match {
		return "punct"
	} else if match, _ := regexp.MatchString(`\w+$`, word); match {
		if strings.ToLower(word) == word {
			return "downcase"
		} else if strings.Title(word) == word {
			return "upcase"
		} else {
			return "mixedcase"
		}
	}
	return "other"
}

func simplePOS(pos string) string {
	if strings.HasPrefix(pos, "V") {
		return "v"
	}
	return strings.Split(pos, "-")[0]
}

func encode(corpus featureSet) *binaryMaxentClassifier {
	mapping := make(map[string]int) // maps (fname-fval-label) -> fid
	count := make(map[string]int)   // maps (fname, fval) -> count
	weights := []float64{}

	labels := []string{}
	for _, entry := range corpus {
		label := entry.label
		if !stringInSlice(label, labels) {
			labels = append(labels, label)
		}

		for i, fname := range featureOrder {
			fval := entry.features[i]
			key := strings.Join([]string{fname, fval}, "-")
			count[key]++
			entry := strings.Join([]string{fname, fval, label}, "-")
			if _, found := mapping[entry]; !found {
				mapping[entry] = len(mapping)
			}

		}
	}
	return newMaxentClassifier(weights, mapping, labels)
}

func empiricalCount(corpus featureSet, encoding *binaryMaxentClassifier) *mat.VecDense {
	count := mat.NewVecDense(len(encoding.mapping)+1, nil)
	for _, entry := range corpus {
		for _, encoded := range encoding.encodeGIS(entry.features, entry.label) {
			idx := encoded.key
			count.SetVec(idx, count.AtVec(idx)+float64(encoded.value))
		}
	}
	return count
}

func addLogs(x, y float64) float64 {
	if x < y+maxLogDiff {
		return y
	} else if y < x+maxLogDiff {
		return x
	}
	base := math.Min(x, y)
	return base + math.Log2(math.Pow(2, x-base)+math.Pow(2, y-base))
}

func sumLogs(logs []float64) float64 {
	if len(logs) == 0 {
		return math.Inf(-1)
	}
	sum := logs[0]
	for _, log := range logs[1:] {
		sum = addLogs(sum, log)
	}
	return sum
}
