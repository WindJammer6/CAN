import os
import numpy as np
import matplotlib.pyplot as plt

PLOT_CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'none']

WM811K_ID_TO_STR = {
	0: 'Center',
	1: 'Donut',
	2: 'Edge-Loc',
	3: 'Edge-Ring',
	4: 'Loc',
	5: 'Random',
	6: 'Scratch',
	7: 'Near-full',
	8: 'none',
}

MIXEDWM38_ID_TO_STR = {
	0: 'Center',
	1: 'Donut',
	2: 'Edge-Loc',
	3: 'Edge-Ring',
	4: 'Loc',
	5: 'Random',      
	6: 'Scratch',
	7: 'Near-full',   
	8: 'none',
}

def counts_by_class(labels, id_to_str, class_order):
	counts = dict.fromkeys(class_order, 0)
	unique, cnts = np.unique(labels, return_counts=True)
	for cls_id, count in zip(unique, cnts):
		name = id_to_str.get(int(cls_id))
		if name in counts:
			counts[name] += int(count)
	return np.array([counts[c] for c in class_order], dtype=np.int64)

def main():
	root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	wm_path = os.path.join(root, 'wm811k-64.npz')
	mw_path = os.path.join(root, 'mixed2m38-64-single.npz')

	wm = np.load(wm_path)
	mw = np.load(mw_path)

	wm_counts = counts_by_class(wm['label'], WM811K_ID_TO_STR, PLOT_CLASSES)
	mw_counts = counts_by_class(mw['label'], MIXEDWM38_ID_TO_STR, PLOT_CLASSES)

	wm_prop = wm_counts / wm_counts.sum()
	mw_prop = mw_counts / mw_counts.sum()

	x = np.arange(len(PLOT_CLASSES))
	w = 0.4
	plt.figure(figsize=(8, 4))
	plt.bar(x - w/2, wm_prop, width=w, label='WM-811K')
	plt.bar(x + w/2, mw_prop, width=w, label='MixedWM38')
	plt.xticks(x, PLOT_CLASSES, rotation=25)
	plt.ylabel('Proportion')
	plt.legend()
	plt.tight_layout()
	out_path = os.path.join(os.path.dirname(__file__), 'dataset_proportions.png')
	plt.savefig(out_path, dpi=200)
	print(f"Saved plot to: {out_path}")

if __name__ == '__main__':
	main()