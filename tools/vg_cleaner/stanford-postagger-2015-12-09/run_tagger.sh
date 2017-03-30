#!/bin/sh
java -mx300m -cp stanford-postagger.jar:slf4j-api-1.7.21.jar edu.stanford.nlp.tagger.maxent.MaxentTaggerServer -model models/english-bidirectional-distsim.tagger -port 2020
