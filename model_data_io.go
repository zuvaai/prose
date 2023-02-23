package prose

import (
	"bytes"
	"embed"
	"encoding/gob"
	"fmt"
	"path"
)

//go:embed model
var assets embed.FS

const datadir = "model"

// ReadBytes reads an embedded file into a byte slice.
func ReadBytes(filename string) ([]byte, error) {
	return assets.ReadFile(path.Join(datadir, filename))
}

// ReadAndDecodeBytes reads an embedded file into a gob decoder
func ReadAndDecodeBytes(filename string) (*gob.Decoder, error) {
	b, err := ReadBytes(filename)
	if err != nil {
		return nil, fmt.Errorf("unable to read file %s: %w", filename, err)
	}
	return gob.NewDecoder(bytes.NewReader(b)), nil
}
