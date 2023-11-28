import random

test_data = [
    { 'x': 0.25, 'y': 13.91, },
    { 'x': 3.52, 'y': 44.98, },
    { 'x': 5.36, 'y': 3409.53, },
    { 'x': 8.45, 'y': 3276.44, },
    { 'x': 9.89, 'y': 3870.28, },
    { 'x': 10.44, 'y': 5006.14, },
    { 'x': 11.14, 'y': 4237.16, },
    { 'x': 12.42, 'y': 5254.02, },
    { 'x': 16.51, 'y': 6082.87, },
    { 'x': 19.65, 'y': 6926.12 },
]

w = random.randint(0, 500)
b = random.randint(0, 500)
m = len(test_data)
alpha = 0.05

print(f"first weight: {w}. first balance: {b}")

def getLinearValue(x):
    return w * x + b

def getCost():
    def getSquaredError(x,y):
        return (getLinearValue(x) - y)**2

    return sum([getSquaredError(*d.values()) for d in test_data]) / (2 * m)

def getDerivativeCost(isWeight):
    def getSingleCost(x, y):
        error = (getLinearValue(x) - y)
        return error * x if isWeight else error

    return sum([getSingleCost(d['x'], d['y']) for d in test_data])


print(f"initial cost: {getCost()}")

# number of gradient descent iterations
for i in range(100):
    newW = w - (getDerivativeCost(True) * alpha / m)
    newB = b - (getDerivativeCost(False) * alpha / m)
    w = newW
    b = newB

print(f"cost after gradient descent: {getCost()}")
