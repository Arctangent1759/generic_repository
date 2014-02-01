from scipy.io import loadmat
import os, sys
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

def extract_features(images,labels):

    num_data=len(labels)
    image_width = images.shape[0]
    image_height = images.shape[1]

    example_list=[]
    label_list=[]
    for i in range(num_data):
        features=[]
        for y in range(image_height):
            for x in range(image_width):
                features.append(images[y][x][i])
        example_list.append(features)
        label_list.append(labels[i])
    return example_list, label_list


def train_digit_classifier(example_list,label_list):
    prob = problem(label_list,example_list)
    param = parameter('-B 10000 -c 0.001 -s 2')
    return train(prob,param)

def testModel(model, example_list, label_list):
    p_label, p_acc, p_val = predict(label_list, example_list, model)
    accuracy, mean_squared_error, squared_correlation_coefficient = evaluations(label_list, p_label)
    return accuracy

def main(args):
    if len(args)==2 and args[1]=="plot":
        if hasMatPlotLib:
            error_vals=[]
            n_example_vals=[]

            data = loadmat('data/train_small.mat')['train'][0]
            test_images, test_labels = loadmat('data/test.mat')['test'][0][0]
            test_example_list, test_label_list = extract_features(test_images,test_labels)

            for n in range(len(data)):
                images, labels = data[n][0][0]
                example_list, label_list = extract_features(images,labels)
                m=train_digit_classifier(example_list,label_list)
                error_vals.append(100.0 - testModel(m,test_example_list,test_label_list))
                n_example_vals.append(len(labels))

            plt.plot(n_example_vals,error_vals)
            plt.ylabel('Error Rate on Test Set')
            plt.xlabel('Training Set Size')
            plt.title(r'Effect of Training Set Size on Error Rate')
            plt.show()
        else:
            print "matplotlib was not found. Please install matplotlib and try again."
            exit(1)
    elif len(args)==2 or (len(args)==3 and args[1] in ['0','1','2','3','4','5','6']):
        if len(args)==3:
            images, labels = loadmat('data/train_small.mat')['train'][0][int(args[1])][0][0]
        else:
            images, labels = loadmat('data/train_small.mat')['train'][0][6][0][0]
        test_images, test_labels = loadmat('data/test.mat')['test'][0][0]
        test_example_list, test_label_list = extract_features(test_images,test_labels)
        example_list, label_list = extract_features(images,labels)
        m=train_digit_classifier(example_list,label_list)
        testModel(m,test_example_list,test_label_list)
        save_model(args[2 if len(args)==3 else 1],m)
    else:
        print """
Usage: 
To train a classifier with the maximum number of training data:
python q1.py <filename>

To train a classifier with the a specific number of training data, specify the index in data/train_small.mat in <index>:
python q1.py <index> <filename>

Where the following determines the correspondence between <index> and number of samples:
Index \t Samples
0 \t 100
1 \t 200
2 \t 500
3 \t 1000
4 \t 2000
5 \t 5000
6 \t 10000

To plot error with respect to number of training data (requires an installation of matplotlib):
python q1.py plot
        """


if __name__=="__main__":
    main(sys.argv)
