package engine

func u8Max(a uint8, b uint8) uint8 {
	if a > b {
		return a
	}
	return b
}

func abs(v int16) int16 {
	if v < 0 {
		return -v
	}
	return v
}

func i16min(a int16, b int16) int16 {
	if a < b {
		return a
	}
	return b
}

func i16max(a int16, b int16) int16 {
	if a > b {
		return a
	}
	return b
}
