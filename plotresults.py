import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_points(plottable, ylabel=""):
	fig = plt.figure()
	for ii, test_set in enumerate(plottable.keys()):
		categories = plottable[test_set].keys()
		ax = fig.add_subplot(2,2,ii)
		ax.set_title("tested on "+test_set)
		ax.set_xlabel("Rand error")
		ax.set_ylabel(ylabel)

		ax.grid(True,linestyle='-',color='0.75')
		x, y = two_dimensionalize(plottable[test_set])
		ax.scatter(x, y, marker='o')
	plt.show()

def plot_boxplots(plottable, title=""):
	fig = plt.figure()
	for ii, test_set in enumerate(plottable.keys()):
		# categories = plottable[test_set].keys()
		ax = fig.add_subplot(2,2,map_name_to_subfig(test_set))
		ax.boxplot([v for k, v in plottable[test_set].iteritems()], vert=False, patch_artist=True)
		keys = [abbreviate_feature_name(k) for k, v in plottable[test_set].iteritems()]
		ax.set_title("tested on "+test_set)
		ax.set_xlabel("Rand error")
		plt.yticks(np.arange(1,1+len(plottable[test_set])), keys)

	fig.suptitle(title, fontsize=24)
	plt.show()

def plot_bar_plots(keyed_values, title="", xlabel="", ylabel=""):
	fig = plt.figure()
	values = [v for k,v in keyed_values.iteritems()]
	labels = [abbreviate_feature_name(k) for k,v in keyed_values.iteritems()]
	positions = np.arange(len(values))
	p1 = plt.bar(positions, values)
	plt.xticks(positions+0.4, labels)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()


def map_name_to_subfig(name):
	if name == "topleft": return 1
	if name == "topright": return 2
	if name == "bottomleft": return 3
	if name == "bottomright": return 4
	raise KeyError("Unknown subfig name: %s" % (name))

def abbreviate_feature_name(name, feature_delimiter=",", component_delimiter=": "):
	abbrev = []
	for component in name.split("+"):
		label = []
		for feature in component.split("AND"): label.append(feature[0])
		abbrev.append(feature_delimiter.join(label))
	return component_delimiter.join(abbrev)



def two_dimensionalize(categorized):
	points_x = []
	points_y = []
	for ii, label in enumerate(categorized.keys()):
		for value in categorized[label]:
			points_x.append(value)
			points_y.append(ii)
	return points_x, points_y

def compare_features(results, comparison="features", features=None, cues=None, skip_train=True):
	test_sets = results.keys()
	plottable = {}
	for test_set in test_sets:
		train_sets = results[test_set].keys()
		plottable[test_set] = {}
		if features == None:
			features = results[test_set][train_sets[0]].keys()
		if cues == None:
			cues = results[test_set][train_sets[0]][features[0]]
		if comparison == "features":
			for feature in features: plottable[test_set][feature] = []
		elif comparison == "cues":
			for cue in cues: plottable[test_set][cue] = []
		elif comparison == "features+cues" or comparison == "cues+features":
			for feature in features:
				for cue in cues:
					plottable[test_set][feature+"+"+cue] = []
		else: raise KeyError("Unknown comparison: %s" % (comparison))
		for train_set in train_sets:
			if train_set == test_set and skip_train: continue
			for feature in features:
				for cue in cues:
					if comparison == "features": keystr = feature
					elif comparison == "cues": keystr = cue
					elif comparison == "features+cues" or comparison == "cues+features":
						keystr = feature+"+"+cue
					plottable[test_set][keystr].append(
						results[test_set][train_set][feature][cue])
	return plottable

def compare_to_base(results, base_feature="base", base_cue="idsia"):
	wins = {}
	totals = {}
	for test_set, d1 in results.iteritems():
		for train_set, d2 in d1.iteritems():
			for feature, d3 in d2.iteritems():
				for cue, d4 in d3.iteritems():
					keystr = feature+"+"+cue
					if keystr not in totals: totals[keystr] = 0
					totals[keystr] += 1
					if keystr not in wins: wins[keystr] = 0
					if (results[test_set][train_set][feature][cue] >
					   results[test_set][train_set][base_feature][base_cue]):
					   wins[keystr] += 1
	rates = {}
	for key in totals.keys():
		rates[key] = float(wins[key])/totals[key]
	return rates

def load_json_file(path):
    f = file(path, "r")
    r = json.load(f)
    f.close()
    return r

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=str,
                    help="Path to the json_file")
    args = parser.parse_args()
    results = load_json_file(args.results_json)
    plot_bar_plots(compare_to_base(results), "Rate of outperforming base features",
    		"feature set", "fraction of train set/test set combinations\n in which specified feature set outperforms base features")
    for poi in ["cues+features", "features", "cues"]:
    	plot_boxplots(compare_features(results, comparison=poi), "Comparison of "+poi)

    
if __name__ == '__main__':
    main()