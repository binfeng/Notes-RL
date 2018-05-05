from sklearn import tree
features=[[140,0],[130,0],[150,1],[170,1]]
labels=["apple","apple","orange","orange"]
clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)
print (clf.predict([[150,1]]))
