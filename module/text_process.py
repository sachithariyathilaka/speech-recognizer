class TextProcess:
    def __init__(self):
        char_map_str = """
		' 0
		<SPACE> 1
		a 2
		b 3
		c 4
		d 5
		e 6
		f 7
		g 8
		h 9
		i 10
		j 11
		k 12
		l 13
		m 14
		n 15
		o 16
		p 17
		q 18
		r 19
		s 20
		t 21
		u 22
		v 23
		w 24
		x 25
		y 26
		z 27
		A 28
		B 29
		C 30
		D 31
		E 32
		F 33
		G 34
		H 35
		I 36
		J 37
		K 38
		L 39
		M 40
		N 41
		O 42
		P 43
		Q 44
		R 45
		S 46
		T 47
		U 48
		V 49
		W 50
		X 51
		Y 52
		Z 53
		. 54
		- 55
		, 56
		? 57
		; 58
		! 59
		"""
        self.char_map = {}
        self.index_map = {}

        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int_sequence(self, text):
        int_sequence = []

        for c in text:

            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]

            int_sequence.append(ch)

        return int_sequence

    def int_to_text_sequence(self, labels):
        string = []

        for i in labels:
            string.append(self.index_map[i])

        return ''.join(string).replace('<SPACE>', ' ')
