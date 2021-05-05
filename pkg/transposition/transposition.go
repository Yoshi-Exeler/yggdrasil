package transposition

import (
	"sync"
)

type Table struct {
	Table *sync.Map // The Actual Hashmap holding the Transpositions
}

// Query will perform a lookup in the transposition table and return
// the appropriate entry or nil, if no such entry was found
func (t *Table) Query(hash uint64) *Entry {
	// Query the table
	v, ok := t.Table.Load(hash)
	if !ok {
		return nil
	}
	// type assert to Entry
	ret, ok := v.(Entry)
	if !ok {
		return nil
	}
	return &ret
}

// Commit will add an entry to the transposition table
func (t *Table) Commit(hash uint64, entry Entry) {
	t.Table.Store(hash, entry)
}

type Entry struct {
	Score int16 // The Result of the Search
	Exact bool  // Whether or not the search returned an exact score
	Max   bool  // Whether or not we searched this as a maximizing branch
	Alpha bool  // Whether or not the search returned alpha
	Depth int   // The Depth to which the position was searched
}
