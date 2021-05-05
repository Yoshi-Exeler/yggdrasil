package engine

import (
	"sync"

	chess "github.com/Yoshi-Exeler/chesslib"
)

type PVT struct {
	pv     []*chess.Move
	depth  int
	pvlock sync.Mutex
}

// NewPVT returns a new Prinicpal variation table
func NewPVT() *PVT {
	return &PVT{
		pv:     make([]*chess.Move, 0),
		depth:  0,
		pvlock: sync.Mutex{},
	}
}

// Update will attempt to update the PVT, if the depth is sufficient
func (p *PVT) Update(pv []*chess.Move, depth int) {
	p.pvlock.Lock()
	defer p.pvlock.Unlock()
	if depth > p.depth {
		p.pv = pv
		p.depth = depth
	}
}

// GetPV returns the current pricipal variation
func (p *PVT) GetPV() []*chess.Move {
	return p.pv
}
