from sklearn import tree

# Note: A tree is basically a package that allows you to 
# Implement a decision tree algorithm

#Now to implement our first variable 

#This is basically a list of lists that 
# contain [height, weight, shoe size]
x = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],
[190,90,47],[175,64,39],[177,70,40],[159, 55, 37],
[175,75,42],[181,85,42]]




#This is basically a string list that contains
# The lables that form a relationship 
#with the data provided
y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 
     'female', 'male', 'female', 'male']

# Now we make our classifier 

classifier = tree.DecisionTreeClassifier()

classifier = classifier.fit(x,y)

prediction = classifier.predict([[180, 43, 28]])

print (prediction)

#That's pretty cool, it just predicted that it is male

# Doubts :- What is a decision tree 
# What is fit ?
# What is DecisionTreeClassifier ?
# What is predict ?
# How does [[180,43,28]] depict a 2 diamentional treee ?



