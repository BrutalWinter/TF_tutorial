import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

############################################################################################################
                #   tf.function   #
# https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/
# https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md#functions-that-create-state
# https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/
'''
AutoGraph is one of the most exciting new features of Tensorflow 2.0: it allows transforming a subset of Python syntax into its portable,
high-performance and language agnostic graph representation bridging the gap between Tensorflow 1.x and the 2.0 release based on eager execution.
'''
############################################################################################################
#To use tf.function the first thing to do is to refactor the old 1.x code, wrapping the code we want to execute into a session.
# In general, where in 1.x there was a session execution, now there in 2.x is Python function.
#Note: this is a huge advantage since the software architecture it allows defining is cleaner, and easy to maintain and document.
def f0():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    return y
#What happens now? Nothing. Tensorflow 2.0 works in eager mode by default, this means that we just defined a standard Python function and if we evaluate it:
print(f0().numpy())
#Let’s just add the @tf.function decoration to the f function.
# For the sake of clarity (and to debug in the old-school print driven way) let’s add even a print and a tf.print statement inside the function body:
@tf.function
def f1():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print("PRINT (error-test): ", y)
    tf.print("TF-PRINT (error-test): ", y)
    return y
f1()
#State (like tf.Variable objects) are only created the first time the function f is called.
# add a new layer will produce tf.variables.
############################################################################################################
                                   # Object Oriented solution (recommended):#
'''
Converting a function that works in eager mode to its Graph representation requires to think about the Graph even though we are working in eager mode.

1.Declare f as a function that accepts an input parameter: the parameter can be a tf.Variable or any other input type.
2.Create a function that inherits the Python variable from the parent scope, and check in the function body if it has already been declared (if b != None).
3.Wrap everything inside a class. The __call__ method is the function we want to execute and the variable is declared as a private attribute (self._b).
The same declaration check of point 2 has to be performed. In practice, this is the Object Oriented solution that is functionally equivalent to the one suggested in point 2.
'''
############################################################################################################
######### The ugly solution with global variables (highly discouraged):
b = None
@tf.function
def f2():
    a = tf.constant([[10, 10], [11., 1.]])
    x = tf.constant([[1., 0.], [0., 1.]])
    global b
    if b is None:
        b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y
f2()


class F():
    def __init__(self):
        self._b = None

    @tf.function
    def __call__(self):
        a = tf.constant([[10, 10], [11., 1.]])
        x = tf.constant([[1., 0.], [0., 1.]])
        if self._b is None:
            self._b = tf.Variable(12.)
        y = tf.matmul(a, x) + self._b
        print("PRINT: ", y)
        tf.print("TF-PRINT: ", y)
        return y

fun = F()
fun()
'''
When defining a function you want to accelerate converting it to its graph representation, you have to define its body thinking about the Graph is being built.
There is no 1:1 match between eager execution and the graph built by @tf.function;
thanks to AutoGraph there is no need to worry about the order of the operation execution,
but special attention is required when definition function with objects that can create a state (tf.Variable).
'''
###A second option to solve the problem is to move the variable outside the function body.
@tf.function
def f3(b):
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y

b1 = tf.Variable(12.)
f3(b1)


a = tf.Variable(0)
@tf.function
def g0(x):
    x.assign_add(1)
    return x

print(g0(a))
print(g0(a))
print(g0(a))
print('################################\n'*5)

############################################################################################################
                #   part2   #
'''
the eager version to its graph representation and faced the problems that arise when working with functions that create a state.
'''
############################################################################################################

@tf.function
def f(x):
    print("Python execution: ", x)
    tf.print("Graph execution: ", x)
    return x
# '''
# Line 1: the function accepts a Python variable x that can be literally everything.
# Line 2: the print function is executed once, only during the function creation.
# Line 3: the tf.print function is executed every time the graph is evaluated.
# Line 4: x is returned.
# '''
print("##### float32 test #####")
a = tf.constant(1, dtype=tf.float32)
print("first call")
f(a)
a = tf.constant(1.1, dtype=tf.float32)
print("second call")
f(a)

print("##### uint8 test #####")

b = tf.constant(2, dtype=tf.uint8)
print("first call")
f(b)
print(tf.autograph.to_code(f.python_function))
# print(tf.autograph.to_code(f.python_function))
b = tf.constant(3, dtype=tf.uint8)
print("second call")
f(b)
#A graph is created for every different input type of the tf.Tensor object passed.
# We can have a look at the graph version of the function f by using the tf.autograph module.
print(tf.autograph.to_code(f.python_function))


print('################################\n'*5)
########################
def printinfo(x):
  print("Type: ", type(x), " value: ", x)

print("##### int test #####")
print("first call")
a = 1
printinfo(a)
f(a)
print("second call")
b = 2
printinfo(b)
f(b)

print("##### float test #####")
print("first call")
a = 1.0
printinfo(a)
f(a)
print("second call")
b = 2.0
printinfo(b)
f(b)

print("##### complex test #####")
print("first call")
a = complex(1.0, 2.0)
printinfo(a)
f(a)
print("second call")
b = complex(2.0, 1.0)
printinfo(b)
f(b)

ret = f(1.0)
if tf.float32 == ret.dtype:
    print("f(1.0) returns float")
else:
    print("f(1.0) return ", ret)

#########################
#Performance measurement The following code is a simple benchmark to check if the previous reasoning is correct.

@tf.function
def g(x):
  return x

start = time.time()
for i in tf.range(1000):
  g(i)
end = time.time()

print("tf.Tensor time elapsed: ", (end-start))

start = time.time()
for i in range(1000):
  g(i)
end = time.time()

print("Native type time elapsed: ", (end-start))
#Conclusion: do use tf.Tensor everywhere!!!!!!!!!!!!!!!!
#AutoGraph is highly optimized and works well when the input is a tf.Tensor object,
# while it creates a new graph for every different input parameter value with a huge drop in performance.







