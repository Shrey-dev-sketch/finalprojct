import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from preprocess import preprocess
text = headline + body
nlp = spacy.load('en_core_web_sm')
text_pos = ' '.join([token.pos_ for token in nlp(text)])
lexicon = Empath()
semantic_arr = np.asarray([value for value in lexicon.analyze(text,normalize = False).values()])
categories = [key for key in lexicon.analyze("").keys()]
semantics = " ".join(categories[j] for j in range(len(semantic_arr)))
x = sp.csr_matrix(preprocess(text,text_pos, semantics),(1,20449))
x_pred = model.predict(x)



