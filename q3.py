from scipy.io import loadmat
from liblinearutil import *
import matplotlib.pyplot as plt
import sys

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


def train_digit_classifier(example_list,label_list,b,c):
    prob = problem(label_list,example_list)
    param = parameter('-B {0} -s 2 -c {1}'.format(str(b),str(c)))
    return train(prob,param)

def testModel(model, example_list, label_list):
    p_label, p_acc, p_val = predict(label_list, example_list, model)
    accuracy, mean_squared_error, squared_correlation_coefficient = evaluations(label_list, p_label)
    return accuracy


def splitArr(a,k):
    arr=[[] for i in range(k)]
    for i in range(len(a)):
        arr[i%k].append(a[i])
    return arr


K=10

def main(args):
    print ("\n :: Loading data... :: \n")
    raw_data, raw_labels = loadmat('data/train_small.mat')['train'][0][6][0][0]
    test_images, test_labels = loadmat('data/test.mat')['test'][0][0]
    example_list, label_list = extract_features(raw_data,raw_labels)
    test_example_list, test_label_list = extract_features(test_images,test_labels)

    print ("\n :: Splitting data into {0} partitions... :: \n".format(K))
    feature_partitions = splitArr(example_list,K)
    label_partitions = splitArr(label_list,K)

    classifiers=[]

    print ("\n :: Beginning cross validation... :: \n".format(K))
    for c in [0.01,.001,0.0001]:
        for b in [100, 1000, 10000]:
            totalError=0.0
            for i in range(K):
                print "\n :: Training classifier for c={0} b={1} partition {2} :: \n".format(c,b,i)
                curr_partition_features = reduce(lambda x,y:x+y,feature_partitions[:i] + feature_partitions[i+1:])
                curr_partition_labels = reduce(lambda x,y:x+y,label_partitions[:i] + label_partitions[i+1:])

                test_partition_features = feature_partitions[i]
                test_partition_labels = label_partitions[i]

                m=train_digit_classifier(curr_partition_features,curr_partition_labels,b,c)

                print "\n :: Testing classifier for c={0} b={1} partition {2} :: \n".format(c,b,i)
                error = 100.0-testModel(m,test_partition_features,test_partition_labels)
                print "\n :: Classifier for c={0} b={1} partition {2} error is {3} :: \n".format(c,b,i,error)
                totalError += error
            averageError = totalError/K
            print "\n :: Average error for c={0} b={1} is {2} :: \n".format(c,b,averageError)
            classifiers.append((averageError,(b,c)))

    b,c=min(classifiers)[1]
    print "\n :: Training classifier with all data with best b={0} c={1} :: \n".format(b,c)

    m=train_digit_classifier(example_list,label_list,b,c)
    error = 100.0-testModel(m,test_example_list,test_label_list)
    print "Final error: {0}".format(error);

    save_model(args[1],m)


if __name__=="__main__":
    main(sys.argv)
