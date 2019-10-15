# seqPointer
This a a seq2seq-based model with coverage mechanism for keyphrase extraction

# pointerBaseline
	init Embedding with random
	hidden_size = 500
	without coverage
	n_iteration = 4000		Average  P: 0.7692028985507242; R: 0.5645351025785812; F: 0.6164366704880498

# pointerWithCov
	init Embedding with random
	hidden_size = 500
	with coverage
	n_iteration = 4000		Average  P: 0.7735617039964867; R: 0.5860844469540127; F: 0.6317792513444649

# pointerWithPreCov
	init Embedding with pre-train fasttext from facebook
	hidden_size = 300
	with coverage
	n_iteration = 4000		Average  P: 0.6297760210803689; R: 0.35661924629315905; F: 0.42553703798763
