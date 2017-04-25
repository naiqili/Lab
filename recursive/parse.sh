scriptdir=parser

java -mx150m -cp "$scriptdir/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
 -outputFormat "penn" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz '_data/raw.txt' > '_data/parsed.txt'
