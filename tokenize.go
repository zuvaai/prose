package prose

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

type TokenTester func(string) bool

type Tokenizer interface {
	Tokenize(string) []*Token
}

// iterTokenizer splits a sentence into words.
type iterTokenizer struct {
	specialRE      *regexp.Regexp
	sanitizer      *strings.Replacer
	contractions   []string
	splitCases     []string
	suffixes       []string
	prefixes       []string
	emoticons      map[string]struct{}
	isUnsplittable TokenTester
}

type TokenizerOptFunc func(*iterTokenizer)

// UsingIsUnsplittableFN gives a function that tests whether a token is splittable or not.
func UsingIsUnsplittable(x TokenTester) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.isUnsplittable = x
	}
}

// Use the provided special regex for unsplittable tokens.
func UsingSpecialRE(x *regexp.Regexp) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.specialRE = x
	}
}

// Use the provided sanitizer.
func UsingSanitizer(x *strings.Replacer) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.sanitizer = x
	}
}

// Use the provided suffixes.
func UsingSuffixes(x []string) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.suffixes = x
	}
}

// Use the provided prefixes.
func UsingPrefixes(x []string) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.prefixes = x
	}
}

// Use the provided map of emoticons.
func UsingEmoticons(x map[string]struct{}) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.emoticons = x
	}
}

// Use the provided contractions.
func UsingContractions(x []string) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.contractions = x
	}
}

// Use the provided splitCases.
func UsingSplitCases(x []string) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.splitCases = x
	}
}

// Constructor for default iterTokenizer
func NewIterTokenizer(opts ...TokenizerOptFunc) *iterTokenizer {
	tok := new(iterTokenizer)

	// Set default parameters
	tok.contractions = contractions
	tok.emoticons = emoticons
	tok.isUnsplittable = func(_ string) bool { return false }
	tok.prefixes = prefixes
	tok.sanitizer = sanitizer
	tok.specialRE = internalRE
	tok.suffixes = suffixes

	// Apply options if provided
	for _, applyOpt := range opts {
		applyOpt(tok)
	}

	tok.splitCases = append(tok.splitCases, tok.contractions...)

	return tok
}

func addToken(s string, toks []*Token) []*Token {
	if strings.TrimSpace(s) != "" {
		toks = append(toks, &Token{Text: s})
	}
	return toks
}

func (t *iterTokenizer) isSpecial(token string) bool {
	_, found := t.emoticons[token]
	return found || t.specialRE.MatchString(token) || t.isUnsplittable(token)
}

func (t *iterTokenizer) doSplit(token string) []*Token {
	tokens := []*Token{}
	suffs := []*Token{}

	last := 0
	for token != "" && utf8.RuneCountInString(token) != last {
		if t.isSpecial(token) {
			// We've found a special case (e.g., an emoticon) -- so, we add it as a token without
			// any further processing.
			tokens = addToken(token, tokens)
			break
		}
		last = utf8.RuneCountInString(token)
		lower := strings.ToLower(token)
		if hasAnyPrefix(token, t.prefixes) {
			// Remove prefixes -- e.g., $100 -> [$, 100].
			tokens = addToken(string(token[0]), tokens)
			token = token[1:]
		} else if idx := hasAnyIndex(lower, t.splitCases); idx > -1 {
			// Handle "they'll", "I'll", "Don't", "won't", amount($).
			//
			// they'll -> [they, 'll].
			// don't -> [do, n't].
			// amount($) -> [amount, (, $, )].
			tokens = addToken(token[:idx], tokens)
			token = token[idx:]
		} else if hasAnySuffix(token, t.suffixes) {
			// Remove suffixes -- e.g., Well) -> [Well, )].
			suffs = append([]*Token{
				{Text: string(token[len(token)-1])}},
				suffs...)
			token = token[:len(token)-1]
		} else {
			tokens = addToken(token, tokens)
		}
	}

	return append(tokens, suffs...)
}

// tokenize splits a sentence into a slice of words.
func (t *iterTokenizer) Tokenize(text string) []*Token {
	var tokens []*Token

	clean, white := t.sanitizer.Replace(text), false
	length := len(clean)

	start, index := 0, 0
	cache := map[string][]*Token{}
	for index <= length {
		uc, size := utf8.DecodeRuneInString(clean[index:])
		if size == 0 {
			break
		} else if index == 0 {
			white = unicode.IsSpace(uc)
		}
		if unicode.IsSpace(uc) != white {
			if start < index {
				span := clean[start:index]
				if toks, found := cache[span]; found {
					tokens = append(tokens, toks...)
				} else {
					toks := t.doSplit(span)
					cache[span] = toks
					tokens = append(tokens, toks...)
				}
			}
			if uc == ' ' {
				start = index + 1
			} else {
				start = index
			}
			white = !white
		}
		index += size
	}

	if start < index {
		tokens = append(tokens, t.doSplit(clean[start:index])...)
	}

	return tokens
}

var internalRE = regexp.MustCompile(`^(?:[A-Za-z]\.){2,}$|^[A-Z][a-z]{1,2}\.$`)
var sanitizer = strings.NewReplacer(
	"\u201c", `"`,
	"\u201d", `"`,
	"\u2018", "'",
	"\u2019", "'",
	"&rsquo;", "'")
var contractions = []string{"'ll", "'s", "'re", "'m", "n't"}
var suffixes = []string{",", ")", `"`, "]", "!", ";", ".", "?", ":", "'"}
var prefixes = []string{"$", "(", `"`, "["}
var emoticons = map[string]struct{}{
	"(-8":         struct{}{},
	"(-;":         struct{}{},
	"(-_-)":       struct{}{},
	"(._.)":       struct{}{},
	"(:":          struct{}{},
	"(=":          struct{}{},
	"(o:":         struct{}{},
	"(¬_¬)":       struct{}{},
	"(ಠ_ಠ)":       struct{}{},
	"(╯°□°）╯︵┻━┻": struct{}{},
	"-__-":        struct{}{},
	"8-)":         struct{}{},
	"8-D":         struct{}{},
	"8D":          struct{}{},
	":(":          struct{}{},
	":((":         struct{}{},
	":(((":        struct{}{},
	":()":         struct{}{},
	":)))":        struct{}{},
	":-)":         struct{}{},
	":-))":        struct{}{},
	":-)))":       struct{}{},
	":-*":         struct{}{},
	":-/":         struct{}{},
	":-X":         struct{}{},
	":-]":         struct{}{},
	":-o":         struct{}{},
	":-p":         struct{}{},
	":-x":         struct{}{},
	":-|":         struct{}{},
	":-}":         struct{}{},
	":0":          struct{}{},
	":3":          struct{}{},
	":P":          struct{}{},
	":]":          struct{}{},
	":`(":         struct{}{},
	":`)":         struct{}{},
	":`-(":        struct{}{},
	":o":          struct{}{},
	":o)":         struct{}{},
	"=(":          struct{}{},
	"=)":          struct{}{},
	"=D":          struct{}{},
	"=|":          struct{}{},
	"@_@":         struct{}{},
	"O.o":         struct{}{},
	"O_o":         struct{}{},
	"V_V":         struct{}{},
	"XDD":         struct{}{},
	"[-:":         struct{}{},
	"^___^":       struct{}{},
	"o_0":         struct{}{},
	"o_O":         struct{}{},
	"o_o":         struct{}{},
	"v_v":         struct{}{},
	"xD":          struct{}{},
	"xDD":         struct{}{},
	"¯\\(ツ)/¯":    struct{}{},
}
