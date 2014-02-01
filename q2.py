from scipy.io import loadmat
import os, sys
try:
    from liblinearutil import *
except ImportError:
    lib_path = os.path.abspath('./liblinear')
    sys.path.append(lib_path)
    lib_path = os.path.abspath('./liblinear/python')
    sys.path.append(lib_path)
    from liblinearutil import *
try:
    import matplotlib.pyplot as plt
    hasMatPlotLib=True
except ImportError:
    hasMatPlotLib=False
from q1 import extract_features

if not hasMatPlotLib:
    print "matplotlib was not found. Please install matplotlib and try again."
    exit(1)

def make_confusion(predictions, actual, size):
    assert len(predictions) == len(actual)
    holdit = []
    for i in range(size):
        you_have_defied_the_law = []
        for j in range(size):
            you_have_defied_the_law.append(0)
        holdit.append(you_have_defied_the_law)
    for i in xrange(len(predictions)):
        x, y = int(predictions[i]), int(actual[i])
        holdit[y][x] += 1
    return holdit

if len(sys.argv) == 1:
    m = load_model("q1_svm.model")
else:
    m = load_model(sys.argv[1])

test_images, test_labels = loadmat('data/test.mat')['test'][0][0]
test_example_list, test_label_list = extract_features(test_images, test_labels)

p_label, p_acc, p_vals = predict(test_label_list, test_example_list, m)

predictions = p_label
actual = [x[0] for x in test_label_list]

h = make_confusion(predictions, actual, 10)
plt.imshow(h, interpolation='nearest')
for i, j in enumerate(h):
    for j, c in enumerate(j):
        plt.text(j-.25, i+.2, c, fontsize=12)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(r'Confusion Matrix')
plt.show()
