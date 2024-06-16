import matplotlib.pyplot as plt


def draw():
    x = [55, 64, 73, 82, 91]
    accuracy = [91.7582, 90.2778, 94.3925, 96.1538, 93.7500]
    precision = [92.6893, 90.7566, 95.0302, 97.1717, 95.1515]
    recall = [91.8286, 90.5588, 94.0978, 96.3925, 93.9394]

    plt.figure(figsize=(10, 6))
    plt.plot(x, accuracy, label="Accuracy", color="blue")
    plt.plot(x, precision, label="Precision", color="red")
    plt.plot(x, recall, label="Recall", color="green")
    plt.title('SVM results')
    plt.xlabel('proportion')
    plt.ylabel('indicators')
    plt.legend()
    plt.savefig('./images/result.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    draw()
