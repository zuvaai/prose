package prose

import (
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
)

// A Model holds the structures and data used internally by prose.
type Model struct {
	Name string

	tagger    *PerceptronTagger
	extracter *entityExtracter
}

// DataSource provides training data to a Model.
type DataSource func(model *Model)

// UsingEntities creates a NER from labeled data.
func UsingEntities(data []EntityContext) DataSource {
	return UsingEntitiesAndTokenizer(data, NewIterTokenizer())
}

// UsingEntities creates a NER from labeled data and custom tokenizer.
func UsingEntitiesAndTokenizer(data []EntityContext, tokenizer Tokenizer) DataSource {
	return func(model *Model) {
		corpus := makeCorpus(data, model.tagger, tokenizer)
		model.extracter = extracterFromData(corpus)
	}
}

// LabeledEntity represents an externally-labeled named-entity.
type LabeledEntity struct {
	Start int
	End   int
	Label string
}

// EntityContext represents text containing named-entities.
type EntityContext struct {
	// Is this is a correct entity?
	//
	// Some annotation software, e.g. Prodigy, include entities "rejected" by
	// its user. This allows us to handle those cases.
	Accept bool

	Spans []LabeledEntity // The entity locations relative to `Text`.
	Text  string          // The sentence containing the entities.
}

// ModelFromData creates a new Model from user-provided training data.
func ModelFromData(name string, sources ...DataSource) (*Model, error) {
	model, err := defaultModel(true, true)
	if err != nil {
		return nil, fmt.Errorf("unable to load default model: %w", err)
	}
	model.Name = name
	for _, source := range sources {
		source(model)
	}
	return model, nil
}

// ModelFromDisk loads a Model from the user-provided location.
func ModelFromDisk(path string) (*Model, error) {
	filesys := os.DirFS(path)
	tagger, err := NewPerceptronTagger()
	if err != nil {
		return nil, fmt.Errorf("unable to load POS tager from disk: %w", err)
	}
	classifier, err := loadClassifier(filesys)
	if err != nil {
		return nil, fmt.Errorf("unable to load classifier from disk: %w", err)
	}
	return &Model{
		Name: filepath.Base(path),

		extracter: classifier,
		tagger:    tagger,
	}, nil
}

// ModelFromFS loads a model from the
func ModelFromFS(name string, filesys fs.FS) (*Model, error) {
	// Locate a folder matching name within filesys
	var modelFS fs.FS
	err := fs.WalkDir(filesys, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Model located. Exit tree traversal
		if d.Name() == name {
			modelFS, err = fs.Sub(filesys, path)
			if err != nil {
				return err
			}
			return io.EOF
		}

		return nil
	})
	if err != io.EOF {
		return nil, fmt.Errorf("expected EOF but got: %w", err)
	}
	tagger, err := NewPerceptronTagger()
	if err != nil {
		return nil, fmt.Errorf("unable to create POS tagger FS: %w", err)
	}
	classifier, err := loadClassifier(modelFS)
	if err != nil {
		return nil, fmt.Errorf("unable to load classifier from FS: %w", err)
	}
	return &Model{
		Name: name,

		extracter: classifier,
		tagger:    tagger,
	}, nil
}

// Write saves a Model to the user-provided location.
func (m *Model) Write(path string) error {
	err := os.MkdirAll(path, os.ModePerm)
	if err != nil {
		return fmt.Errorf("unable to open directory: %w", err)
	}
	// m.Tagger.model.Marshal(path)
	return m.extracter.model.marshal(path)
}

/* TODO: External taggers
func loadTagger(path string) *perceptronTagger {
	var wts map[string]map[string]float64
	var tags map[string]string
	var classes []string

	loc := filepath.Join(path, "AveragedPerceptron")
	dec := getDiskAsset(filepath.Join(loc, "weights.gob"))
	checkError(dec.Decode(&wts))

	dec = getDiskAsset(filepath.Join(loc, "tags.gob"))
	checkError(dec.Decode(&tags))

	dec = getDiskAsset(filepath.Join(loc, "classes.gob"))
	checkError(dec.Decode(&classes))

	model := newAveragedPerceptron(wts, tags, classes)
	return newTrainedPerceptronTagger(model)
}*/

func loadClassifier(filesys fs.FS) (*entityExtracter, error) {
	var mapping map[string]int
	var weights []float64
	var labels []string

	maxent, err := fs.Sub(filesys, "Maxent")
	if err != nil {
		return nil, fmt.Errorf("unable to open subdirectory Maxent: %w", err)
	}

	file, err := maxent.Open("mapping.gob")
	if err != nil {
		return nil, fmt.Errorf("unable to open mapping.gob: %w", err)
	}

	err = getDiskAsset(file).Decode(&mapping)
	if err != nil {
		return nil, fmt.Errorf("unable to decode mapping: %w", err)
	}

	file, err = maxent.Open("weights.gob")
	if err != nil {
		return nil, fmt.Errorf("unable to open weights.gob: %w", err)
	}
	err = getDiskAsset(file).Decode(&weights)
	if err != nil {
		return nil, fmt.Errorf("unable to decode weights: %w", err)
	}

	file, err = maxent.Open("labels.gob")
	if err != nil {
		return nil, fmt.Errorf("unable to open labels.gob: %w", err)
	}
	err = getDiskAsset(file).Decode(&labels)
	if err != nil {
		return nil, fmt.Errorf("unable to decode labels: %w", err)
	}

	model := newMaxentClassifier(weights, mapping, labels)
	return newTrainedEntityExtracter(model), nil
}

func defaultModel(tagging, classifying bool) (*Model, error) {
	var tagger *PerceptronTagger
	var classifier *entityExtracter
	var err error
	if tagging || classifying {
		tagger, err = NewPerceptronTagger()
		if err != nil {
			return nil, fmt.Errorf("unable to load default POS tagger: %w", err)
		}
	}
	if classifying {
		classifier, err = newEntityExtracter()
		if err != nil {
			return nil, fmt.Errorf("unable to load default NER: %w", err)
		}
	}

	return &Model{
		Name: "en-v2.0.0",

		tagger:    tagger,
		extracter: classifier,
	}, nil
}
