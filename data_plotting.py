import matplotlib.pyplot as plt


def visualize(*args, title=None, method=plt.plot):
    plt.figure(figsize=(10, 5))
    for (xData, yData, name) in args:
        method(xData, yData, label=name)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
