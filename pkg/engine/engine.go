package engine

import (
	"fmt"
	"sort"
	"time"

	chess "rxchess"
	opening "rxchess/opening"
)

/* Possible Optimizations
*  Multithreading = ~5x Performance Increase
*  SimulatePositions with different parents by only going back to the First Common Ancestor ~1.5x Performance Increase
*  Improve Hash table Housekeeping
*  LazySMP --> Search on many threads at the same time with a shared hashtable
*  Endgame Eval and Sequence Database
*  Iterative Deepening in combination with the Hashtable and Pruning
*  Caputre Heuristic Most Valueable Victim Least Valuable Attacker
*  Save Moves that Cause Pruning, Search after Caputres
* https://www.duo.uio.no/bitstream/handle/10852/53769/master.pdf?sequence=1&isAllowed=y
*
 */

var TxReset = uint(0)
var CnReset = 0
var TxGetSeq = uint(0)
var CnGetSeq = 0
var TxSTN = uint(0)
var CnSTN = 0
var TxEval = uint(0)
var CnEval = 0
var MaxQD = 0

// MaxScore is Bigger than the Maximum Score Reachable
const MaxScore = float32(3000)

// MinScore is Smaller than the Minimum Score Reachable
const MinScore = float32(-3000)

// Engine is the Minimax Engine
type Engine struct {
	UseOpeningTheory   bool
	Game               *chess.Game
	ECO                *opening.BookECO
	Origin             chess.Position
	Simulation         *chess.Position
	Root               *Node
	Color              chess.Color
	GeneratedNodes     uint
	EvaluatedNodes     uint
	Visited            uint
	LoadedFromPosCache uint
	FrontierUnstable   uint
	QEvaluatedNodes    uint
	CurrentBestNode    *Node
	EvaluationCache    map[[16]byte]float32
}

// Node is a node in the Engine
type Node struct {
	Parent          *Node
	Depth           uint8
	Value           *chess.Move
	Leaves          []*Node
	LeavesGenerated bool
	StatusEval      float32
	StatusChecked   bool
}

// NewEngine returns a new Engine from the specified game
func NewEngine(g *chess.Game, clr chess.Color) *Engine {
	pos := *g.Position()
	return &Engine{UseOpeningTheory: false, ECO: opening.NewBookECO(), Game: g, Color: clr, Origin: *g.Position(), Simulation: &pos, Root: &Node{Value: nil, Depth: 0}, EvaluationCache: make(map[[16]byte]float32)}
}

// DoPerft Will generate the full tree up to the Given Depth
func (e *Engine) DoPerft(depth uint) {
	// Generate the First Layer
	currentDepth := uint(1)
	currentLayer := e.Root.GenerateLeaves(e)
	fmt.Printf("Layer 1 - %v Nodes\n", len(currentLayer))
	// Generate the Required Layers
	for {
		// Proceed to the next layer
		currentDepth++
		// Init the Next Layer
		nextLayer := []*Node{}
		// For Every Node in the current Layer
		for _, cnode := range currentLayer {
			// Generate the Nodes following this node
			following := cnode.GetLeaves(e)
			// Append the Follwing Nodes to the Next Layer
			nextLayer = append(nextLayer, following...)
		}
		// Append the Layer to the Tree
		fmt.Printf("Layer %v - %v Nodes\n", currentDepth, len(nextLayer))
		// If this Layer is the Lowest Layer we will simulate, Eval All nodes
		if currentDepth == depth {
			break
		}
		// Set Current
		currentLayer = nextLayer
	}
}

// GetOpeningMove returns an opening theory move from ECO if one exists
func (e *Engine) GetOpeningMove() *chess.Move {
	prevMoves := e.Game.Moves()
	moveIndex := len(prevMoves)
	opennings := e.ECO.Possible(e.Game.Moves())
	sort.Sort(byOpeningLength(opennings))
	for _, op := range opennings {
		moves := op.Game().Moves()
		if len(moves) > moveIndex {
			usable := true
			for idx, mv := range prevMoves {
				if moves[idx].String() != mv.String() {
					usable = false
				}
			}
			if usable {
				fmt.Println("[YGG] Playing the", op.Title())
				fmt.Println("[YGG] Next move", moves[moveIndex])
				return moves[moveIndex]
			}
		}
		break
	}
	fmt.Println("[YGG] No move was found in ECO")
	return nil
}

// GetOpeningName returns the current theory name
func (e *Engine) GetOpeningName() string {
	prevMoves := e.Game.Moves()
	moveIndex := len(prevMoves)
	opennings := e.ECO.Possible(e.Game.Moves())
	sort.Sort(byOpeningLength(opennings))
	for _, op := range opennings {
		moves := op.Game().Moves()
		if len(moves) > moveIndex {
			usable := true
			for idx, mv := range prevMoves {
				if moves[idx].String() != mv.String() {
					usable = false
				}
			}
			if usable {
				return op.Title()
			}
		}
		break
	}
	return ""
}

// ByMoveVariance will Sort by Captures and Checks, which have the highest potential score variance
type byMoveVariance []*Node

func (a byMoveVariance) Len() int      { return len(a) }
func (a byMoveVariance) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a byMoveVariance) Less(i, j int) bool {
	if a[i].Value.HasTag(chess.Check) && !a[j].Value.HasTag(chess.Check) {
		return true
	}
	if a[i].Value.HasTag(chess.Capture) && !a[j].Value.HasTag(chess.Capture) {
		return true
	}
	return false
}

type byOpeningLength []*opening.Opening

func (a byOpeningLength) Len() int           { return len(a) }
func (a byOpeningLength) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byOpeningLength) Less(i, j int) bool { return len(a[i].PGN()) > len(a[j].PGN()) }

// ResetStats will reset the Statistics of the Engine
func (e *Engine) ResetStats() {
	e.Visited = 0
	e.GeneratedNodes = 0
	e.EvaluatedNodes = 0
	e.LoadedFromPosCache = 0
}

// Search will Search the Tree using Minimax
func (e *Engine) Search(depth uint) *chess.Move {
	// Check that we can actually play a move here
	if statusIsEnd(e.Origin.Status()) {
		fmt.Println("[YGG] Aborting search, no possible moves exist")
		return nil
	}
	if e.UseOpeningTheory {
		// Check the ECO for an Opening Move
		move := e.GetOpeningMove()
		if move != nil {
			return move
		}
		e.UseOpeningTheory = false
	}
	// Call Minimax
	bestNode, bestScore := e.MinimaxPruning(e.Root, MinScore, MaxScore, depth, true)
	// Get the Origin Move of the Best Leaf
	origin := bestNode.getSequence(e)[0]
	fmt.Printf("[YGG] Sequence:%v\nTarget:%v\nOrigin:%v D:%v\n", bestNode.getSequence(e), bestScore, origin, bestNode.Depth)
	fmt.Printf("[YGG] Generated:%v Visited:%v Evaluated:%v LoadedFromCache:%v\n", e.GeneratedNodes, e.Visited, e.EvaluatedNodes, e.LoadedFromPosCache)
	fmt.Printf("[YGG] FrontierUnstable:%v QuiescEvals:%v\n", e.FrontierUnstable, e.QEvaluatedNodes)

	return origin
}

// GetColor offers the Option to invert colors
func (e *Engine) GetColor(inv bool) chess.Color {
	if inv {
		if e.Color == chess.Black {
			return chess.White
		}
		return chess.Black
	}
	return e.Color
}

// QuiescenceSearch will refine the Score of the Specified Node by extending the Search until a Stable position is found
func (e *Engine) QuiescenceSearch(node *Node, alpha float32, beta float32, inv bool) float32 {
	if node.Depth > uint8(MaxQD) {
		MaxQD = int(node.Depth)
		fmt.Println("DepthMX:", MaxQD)
	}
	e.QEvaluatedNodes++
	e.simulateToNode(node)
	standingPat := EvalStatic(e.Simulation, chess.White)
	if standingPat >= beta {
		return beta
	}
	if alpha < standingPat {
		alpha = standingPat
	}
	// Generate the Unstable Leaves of this node
	unstable := node.GetUnstableLeaves(e)
	for _, nd := range unstable {
		alloc := *nd
		e.simulateToNode(&alloc)
		ev := -e.QuiescenceSearch(&alloc, -beta, -alpha, !inv)
		if ev >= beta {
			return beta
		}
		if ev > alpha {
			alpha = ev
		}
	}
	return alpha
}

// MinimaxPruning will traverse the Tree using the Minimax Algorithm with Alpa/Beta Pruning
func (e *Engine) MinimaxPruning(node *Node, alpha float32, beta float32, depth uint, max bool) (*Node, float32) {
	e.Visited++
	if depth == 0 {
		nseval := e.NodeStatusScore(node)
		if nseval > -1 {
			return node, nseval
		}
		return node, node.Evaluate(e, alpha, beta, false)
	}
	if max {
		best := MinScore
		bestNode := &Node{}
		leaves := node.GetLeaves(e)
		sort.Sort(byMoveVariance(leaves))
		for _, child := range leaves {
			nod, ev := e.MinimaxPruning(child, alpha, beta, depth-1, false)
			if ev > best {
				best = ev
				bestNode = nod
			}
			alpha = f32max(alpha, ev)
			if beta <= alpha {
				break
			}
		}
		return bestNode, best
	}
	worst := MaxScore
	worstNode := &Node{}
	leaves := node.GetLeaves(e)
	sort.Sort(byMoveVariance(leaves))
	for _, child := range leaves {
		nod, ev := e.MinimaxPruning(child, alpha, beta, depth-1, true)
		if ev < worst {
			worst = ev
			worstNode = nod
		}
		beta = f32min(beta, ev)
	}
	return worstNode, worst
}

// Evaluate will return the Evaluation of the Node, using QuiescenseSearch if applicable
func (n *Node) Evaluate(e *Engine, alpha float32, beta float32, inv bool) float32 {
	// Simulate the State of this Node
	e.simulateToNode(n)
	// Hash the Current Position
	posHash := e.Simulation.Hash()
	// Get the Value for this Hash
	storedV := e.EvaluationCache[posHash]
	// If we have a stored value for this Hash return it
	if storedV != 0 {
		e.LoadedFromPosCache++
		return storedV
	}
	score := float32(0)
	// Check wether or not this node is stable
	unstableLeaves := n.GetUnstableLeaves(e)
	// if it is unstable begin Quiescence
	if len(unstableLeaves) > 0 {
		e.FrontierUnstable++
		score = e.QuiescenceSearch(n, alpha, beta, inv)
		// if the node is stable perform Static Evaluation
	} else {
		score = EvalStatic(e.Simulation, e.Color)
	}

	// Save this Evaluation to the Cache
	e.EvaluatedNodes++
	e.EvaluationCache[posHash] = score
	return score
}

// NodeStatusScore Returns the Status evaluation of this node
func (e *Engine) NodeStatusScore(n *Node) float32 {
	// if this is the root dont check
	if n.Value == nil {
		return -1
	}
	// if this has a status
	if n.StatusChecked {
		return n.StatusEval
	}
	e.resetSimulation()
	e.simulateToNode(n)
	n.StatusEval = statusEval(e.Simulation.Status())
	n.StatusChecked = true
	return n.StatusEval
}

// statusIsEnd will return wether or not the specified method ends the game
func statusIsEnd(s chess.Method) bool {
	return (s == chess.Checkmate || s == chess.Stalemate)
}

func statusEval(s chess.Method) float32 {
	if s == chess.Checkmate {
		return 1000.0
	}
	if s == chess.Stalemate {
		return 0
	}
	return -1
}

// reverseSequence will generate the inverse of the Specified Sequence
func reverseSequence(seq []*chess.Move) []*chess.Move {
	for i, j := 0, len(seq)-1; i < j; i, j = i+1, j-1 {
		seq[i], seq[j] = seq[j], seq[i]
	}
	return seq
}

// GetLeaves will return the Leaves of the node and generate them if they dont exist yet
func (n *Node) GetLeaves(e *Engine) []*Node {
	if n.LeavesGenerated {
		return n.Leaves
	}
	return n.GenerateLeaves(e)
}

// IsCheck returns if a node is a checking node
func (n *Node) IsCheck() bool {
	return n.Value.HasTag(chess.Check)
}

// GetUnstableLeaves Returns the list of Immediate Recaptures that became availabe through the last move
func (n *Node) GetUnstableLeaves(e *Engine) []*Node {
	leaves := n.GetLeaves(e)
	unstable := []*Node{}
	for _, nd := range leaves {
		if nd.Value.HasTag(chess.Capture) || nd.Value.HasTag(chess.Check) {
			unstable = append(unstable, nd)
		}
	}
	n.Leaves = unstable
	n.LeavesGenerated = true
	return unstable
}

// GenerateLeaves will generate the leaves of the specified node
func (n *Node) GenerateLeaves(e *Engine) []*Node {
	// Reset the Simulation
	e.resetSimulation()
	// Get the Sequence of the Parent node we were called on
	e.simulateToNode(n)
	// Get the Valid Moves From this Position
	vmoves := e.Simulation.ValidMoves()
	// Initialize node collection
	nds := []*Node{}
	// Convert the Moves to Nodes and add them to the Collection
	for _, mov := range vmoves {
		e.GeneratedNodes++
		nds = append(nds, &Node{Parent: n, Value: mov, Depth: n.Depth + 1})
	}
	// Set the Leaves of MV
	n.Leaves = nds
	return nds
}

func (e *Engine) simulateToNode(n *Node) {
	start := time.Now()
	// Reset the Simulation to the Origin
	e.resetSimulation()
	// If this is the Root node we stop now, nothing todo
	if n.Value == nil {
		return
	}
	// Get the Sequence Required to get to this node
	seq := n.getSequence(e)
	// Apply those move to get to the Current Node
	for _, ms := range seq {
		e.Simulation = e.Simulation.Update((ms))
	}
	end := time.Now()
	TxSTN += uint(end.Sub(start))
	CnSTN++
}

// resetSimulation will reset the Simulated game to the original state
func (e *Engine) resetSimulation() {
	start := time.Now()
	npos := e.Origin
	e.Simulation = &npos
	end := time.Now()
	TxReset += uint(end.Sub(start))
	CnReset++
}

// getSequence will return the full sequence of moves required to get to this node including the move of this node
func (n *Node) getSequence(e *Engine) []*chess.Move {
	start := time.Now()
	seq := make([]*chess.Move, 0)
	cnode := n
	for {
		// Add the Value of the Current Node to the Sequence
		seq = append(seq, cnode.Value)
		// Stop if there is no parent
		if cnode.Parent == nil || cnode.Parent == e.Root {
			break
		}
		// Go to the Parent
		cnode = cnode.Parent
	}
	rev := reverseSequence(seq)
	end := time.Now()
	TxGetSeq += uint(end.Sub(start))
	CnGetSeq++
	return rev
}

// EvalStatic will evaluate a board
func EvalStatic(pos *chess.Position, clr chess.Color) float32 {
	start := time.Now()
	// Calculate the Status of this Position
	status := pos.Status()
	// Calculate the Score of this Position
	score := pos.Board().Evaluate(clr)
	// Check for Checkmate
	if status == chess.Checkmate {
		if pos.Turn() == clr {
			return -1000.0
		}
		return 1000.0
	}
	// Check for Stalemate
	if status == chess.Stalemate {
		return 0
	}
	end := time.Now()
	TxEval += uint(end.Sub(start))
	CnEval++
	return score
}

func f32min(a float32, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func f32max(a float32, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
