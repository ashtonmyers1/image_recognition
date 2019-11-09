import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
# define classifier
clf = svm.SVC(gamma=0.001, C=100)
# define testing set
x, y = digits.data[:-1], digits.target[:-1]
clf.fit(x, y)
# Make prediction and display
print('Prediction:', clf.predict(digits.data)[-1])
# Show the image we are trying to predict
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

