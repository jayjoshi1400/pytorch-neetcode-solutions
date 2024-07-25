# https://neetcode.io/problems/gradient-descent

class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        cur = init
        
        while range(iterations):
            grad = 2*cur
            cur = cur - (learning_rate * grad)
            iterations-=1
        return round(cur,5)
