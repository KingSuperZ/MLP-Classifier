# This block of code contains the data needed for the scikit learn algorithm
Xarray = np.array(X)
yarray = np.array(y)
Xtrain, Xtest, ytrain, ytest = train_test_split(Xarray, yarray, test_size=0.2)
# This is where the algorithm is being implemented
alg = KNeighborsClassifier(n_neighbors=1)
alg.fit(Xtrain,ytrain)
# This is where the predicted values as a result of the algorithm is stored
ypred = alg.predict(Xtest)
# This is simply formatting the data so it is distinct to the reader
# Green: Doesn't Interact, Red: Does Interact
# Dots: Training Data, Crosses: Testing Data
color1 = np.where(ytrain == 0, "green","red")
color2 = np.where(ypred == 0, "green","red")
xcord = Xtrain[:,0]
ycord = Xtrain[:,1]
xcord2 = Xtest[:,0] # Stores the x coordinates
ycord2 = Xtest[:,1]
plt.scatter(xcord,ycord, c = color1)
plt.scatter(xcord2,ycord2, c = color2, marker = "x", s = 50)
plt.axis("equal")
plt.grid()
print(accuracy_score(ytest, ypred))
