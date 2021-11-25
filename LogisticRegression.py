#forward pass
def forPropagate(bias, m, n, X_train, w_curr):
  
  b = bias
  w = w_curr # based on number of columns
  #print("Len w_curr: " + str(w))
  w = w.reshape(n,1)
  Z = np.dot(w.T, X_train.T) + b # (1 x 9) * (9 * 3000)
  A = 1 / (1 + np.exp(-Z)) # sigmoid activtaion
  return A

#backward pass
def backPropagate(A, X_train, y_train):
  y_train = y_train.reshape(1, 3000)
  dz = A - y_train
  db = np.sum(dz) / len(X_train) # gradient for b
  
  dw = np.dot(X_train.T, dz.T) / len(X_train) # (9 x 3000) * (3000 * 1) --> take all x_1 and add them, but each training example row (now a column) should be multipled by same dz since all points from one sample are at same location on the gradient (however sum of each x_1 is summed wth the other x_1)
  
  J = -1 * (np.dot(y_train, np.log(A[0]).T) + np.dot((1-y_train), (1-np.log(A[0]).T)))
  
  print("Cost: " + str(J))
  return [dw, db]
  
def LogisticRegression(alpha):
  w_curr = np.ones(n)
  bias = 0  
  for i in range(100000):
    print(i)
    w_curr = w_curr.reshape(n, 1)
    
    A = forPropagate(bias, m, n, X_train, w_curr)
    backPropResults = backPropagate(A, X_train, y_train)

    #print(w_curr - backPropResults[0])

    w_curr = w_curr - alpha * backPropResults[0]
    

    
    bias = bias - alpha * backPropResults[1]
  print (w_curr)
  return (w_curr)

ans = LogisticRegression(0.00007)
ans
