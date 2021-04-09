package engine

// MaxScore is Bigger than the Maximum Score Reachable
const MaxScore = int16(30000)

// MinScore is Smaller than the Minimum Score Reachable
const MinScore = int16(-30000)

// DeltaMax is the maximum gain in a single move permitted by the evaluation function
const DeltaMax = 97

// Delta is the Safety margin for Delta Pruning
const Delta = 20

const NullReduction = 3

var whitePawnPositionalValuesI16 = [64]int16{
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	-1, -1, -1, -1, -1, -1, -1, -1,
	-2, -2, -2, -2, -2, -2, -2, -2,
	-3, -3, -3, -3, -3, -3, -3, -3,
	-4, -4, -4, -4, -4, -4, -4, -4,
	-5, -5, -5, -5, -5, -5, -5, -5,
	0, 0, 0, 0, 0, 0, 0, 0,
}

var whitePiecePositionalValuesI16 = [64]int16{
	0, 0, 0, 0, 0, 0, 0, 0,
	-1, -1, -1, -1, -1, -1, -1, -1,
	-2, -2, -2, -2, -2, -2, -2, -2,
	-3, -3, -3, -3, -3, -3, -3, -3,
	-4, -4, -4, -4, -4, -4, -4, -4,
	-5, -5, -5, -5, -5, -5, -5, -5,
	-6, -6, -6, -6, -6, -6, -6, -6,
	-7, -7, -7, -7, -7, -7, -7, -7,
}

var blackPawnPositionalValuesI16 = [64]int16{
	0, 0, 0, 0, 0, 0, 0, 0,
	5, 5, 5, 5, 5, 5, 5, 5,
	4, 4, 4, 4, 4, 4, 4, 4,
	3, 3, 3, 3, 3, 3, 3, 3,
	2, 2, 2, 2, 2, 2, 2, 2,
	1, 1, 1, 1, 1, 1, 1, 1,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
}

var blackPiecePositionalValuesI16 = [64]int16{
	7, 7, 7, 7, 7, 7, 7, 7,
	6, 6, 6, 6, 6, 6, 6, 6,
	5, 5, 5, 5, 5, 5, 5, 5,
	4, 4, 4, 4, 4, 4, 4, 4,
	3, 3, 3, 3, 3, 3, 3, 3,
	2, 2, 2, 2, 2, 2, 2, 2,
	1, 1, 1, 1, 1, 1, 1, 1,
	0, 0, 0, 0, 0, 0, 0, 0,
}

var pieceValues = [13]int16{
	0,                          // Nopiece
	0, -90, -50, -31, -30, -10, // White Pieces
	0, 90, 50, 31, 30, 10, // Black Pieces
}
