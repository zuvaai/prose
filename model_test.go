package prose

import (
	"embed"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelFromDisk(t *testing.T) {
	data := filepath.Join(testdata, "PRODUCT")

	model, err := ModelFromDisk(data)
	require.NoError(t, err)
	if model.Name != "PRODUCT" {
		t.Errorf("ModelFromDisk() expected = PRODUCT, got = %v", model.Name)
	}

	temp := filepath.Join(testdata, "temp")
	_ = os.RemoveAll(temp)
	fmt.Println(model.extracter.model.labels)
	fmt.Println(model.extracter.model.weights)
	err = model.Write(temp)
	require.NoError(t, err)
	model, err = ModelFromDisk(temp)
	require.NoError(t, err)
	if model.Name != "temp" {
		t.Errorf("ModelFromDisk() expected = temp, got = %v", model.Name)
	}
}

//go:embed testdata/PRODUCT
var embeddedModel embed.FS

func TestModelFromFS(t *testing.T) {
	err := fs.WalkDir(embeddedModel, ".", func(path string, d fs.DirEntry, err error) error {
		return nil
	})
	assert.NoError(t, err)

	// Load the embedded PRODUCT model
	model, err := ModelFromFS("PRODUCT", embeddedModel)
	require.NoError(t, err)
	if model.Name != "PRODUCT" {
		t.Errorf("ModelFromFS() expected = PRODUCT, got = %v", model.Name)
	}

	doc, err := NewDocument("Windows 10 is an operating system",
		UsingModel(model))

	if err != nil {
		t.Errorf("Failed to create doc with ModelFromFS")
	}

	ents := doc.Entities()

	if len(ents) != 1 {
		t.Fatalf("Expected 1 entity, got %v", ents)
	}

	if ents[0].Text != "Windows 10" {
		t.Errorf("Expected to find entity 'Windows 10' with ModelFromFS, got = %v", ents[0].Text)
	}

	if ents[0].Label != "PRODUCT" {
		t.Errorf("Expected to tab entity with PRODUCT, got = %v", ents[0].Label)
	}
}
