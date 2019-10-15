# seqPointer
This a a seq2seq-pointerNet-based model with coverage mechanism for keyphrase extraction

# introduction
We train a seqPointer model with coverage mechanism for keyphrase extraction.
The encoder is a Bi-GRU with 2 layers which encode the source doc into a context vector, and the docoder is a Uni-GRU with 2 layers with pointer attention(a special attention mechanism) to predict the target keyphrase with coverage mechanism.
The data was provided by xiaomi comanpy. We divided it into train and test dataset with the ratio = 0.9, and the result is below.
As the result with pre-train fasttext from facebook was poor, we queryed our usage with it, and we will update it in future.

# requirements
pytorch		--see https://pytorch.org/get-started/locally/

# preprocess data
## Data form:
	'document' \t 'keyphrase'

## Data requirment:
	1:	both 'document' and 'keyphrase' should be split into word
	2:	the keyphrase should be included in the document

## An example:	
	小姐姐 在 街头 演出 王力宏 竟然 出现 在 现场 网友 太 幸运 了	王力宏

# model settings
see concrete .py file, and modify several datapath and hyperparameters

# model run
python pointerBeseline.py

# model result
## pointerBaseline
	init Embedding with random
	hidden_size = 500
	without coverage
	n_iteration = 4000		Average  P: 0.7692028985507242; R: 0.5645351025785812; F: 0.6164366704880498

## pointerWithCov
	init Embedding with random
	hidden_size = 500
	with coverage
	n_iteration = 4000		Average  P: 0.7735617039964867; R: 0.5860844469540127; F: 0.6317792513444649

## pointerWithPreCov
	init Embedding with pre-train fasttext from facebook
	hidden_size = 300
	with coverage
	n_iteration = 4000		Average  P: 0.6297760210803689; R: 0.35661924629315905; F: 0.42553703798763
