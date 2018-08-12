"""Main script."""

import os 
import gensim
import pandas as pd

from src.build_root_data import build_dataset
import src.build_models as model_utils

# Variables
MODEL_PATH = os.environ['WORD2VEC_PATH']
ROOT_PATH = 'data/raw/roots_celex_monosyllabic.txt'


def main():
	print("Loading words")
	roots_to_syllables = build_dataset()
	print("Loading model")
	model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
	for syllable_component in ['onset', 'nucleus', 'coda']:
	    print("Finding systematicity in {syl} component...".format(syl=syllable_component))
	    X, y, words = model_utils.create_dataset(roots_to_syllables, model, syllable_component)
	    N = len(words)
	    print("Using {num} words".format(num=N))
	    clf = model_utils.get_model('linear_svc')
	    X_reduced = model_utils.reduce_dimensionality(X)
	    df_performance = model_utils.evaluate_classifier_with_cv(
	        X_reduced, y, n_folds=2, clf=clf, shuffled=False)
	    permuted = model_utils.permutation_test(X_reduced, y, clf, 100)
	    df = pd.concat([df_performance, permuted])
	    df['component'] = [syllable_component for _ in range(len(df))]
	    components.append(df)
	return pd.concat(components)


if __name__ == "__main__":
	df = main()
	df.to_csv("data/output/metrics.csv")

