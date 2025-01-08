# import mpmath.libmp


# print(mpmath.libmp.BACKEND)

from z3 import *
import mpmath as mp
from mpmath import *




s2 = Solver()
set_param('parallel.enable', True)
set_option(rational_to_decimal=True)
set_option(precision=1000)
set_param('parallel.enable', True)
s2.set("sat.local_search_threads", 28)
s2.set("sat.threads", 28)
# s2.set("timeout",2000)

# # set_param('q', 'mbqi')
# # set_param('smt.qi.max_depth', 10)
# # set_param('sat.arith.solver', 'simplex')
# # set_param('smt.qi.local_lookahead', True)
# # set_param('smt.qi.quant_caching', True)
# # s2.set('solver', 'smt')

# # s2.set("relevancy",0)
# # s2.set(auto_config=False,relevancy=2)
# # set-option :relevancy 0

# set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)


# xp0, yp0, zp0 = Reals('xp0 yp0 zp0')
# initCubeCon = And(10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1765,100*zp0<=17651)

# s2.add(simplify(initCubeCon))


# count = 0
# mp.pretty = False
# mp.dps = 300

# while(count <100):
#     if(s2.check() ==sat):
        
        
#         m = s2.model()
#         print(m)
#         # sleep(1)
#         # posXp = (eval("m[xp0].numerator_as_long()/m[xp0].denominator_as_long()"))
#         # posYp = (eval("m[yp0].numerator_as_long()/m[yp0].denominator_as_long()"))
#         # posZp = (eval("m[zp0].numerator_as_long()/m[zp0].denominator_as_long()"))
        
#         posXp = m[xp0]
#         posYp = m[yp0]
#         posZp = m[zp0]

#         stringxp0 = m[xp0]
        
        
#         print("posXp = ", posXp,"posYp = ", posYp,"posZp = ", posZp)
#         # posXp = mpf(str(stringxp0)+str(0.0000001))
#         posXp2 = mpf(str(stringxp0))
#         print("posXp2 mpf =", posXp2)

#         print(posXp+0.000001)
#         print(mpf(posXp2)+mpf(0.000001))
 
        
#         notTheCurrentPosCons1 = Or(xp0!= m[xp0], yp0!=m[yp0],zp0!= m[zp0])
#         s2.add(notTheCurrentPosCons1)
#         count+=1
#         print("\n\n")






# print("\n\n----------------------")
# print("mpf also breakig")

xp1 = 0.1000000000000000000000000000000315544362088404722164691426113114491869282574043609201908111572265625
xp2 = mpf('0.1000000000000000000000000000000315544362088404722164691426113114491869282574043609201908111572265625')

print("0.1000000000000000000000000000000315544362088404722164691426113114491869282574043609201908111572265625")
print(xp1+.000001)
print(xp2+.000001)
print(mpf('0.1000000000000000000000000000000315544362088404722164691426113114491869282574043609201908111572265625')+mpf('0.000001'))

# mp.dps = 30


# a = mpf(0.1000000000000000000000000000000315544362088404722164691426113114491869282574043609201908111572265625)
# b = mpf(-0.1)
# c = mpf(-0.1)
# print(a+0.00001)

# x, y = Reals('x y')

# s2.add(x == 0.1000000000000000000000000000000315544362088404722164691426113114491869282574043609201908111572265625)
# s2.add(y == x+0.00001)

# print(s2.check())
# print(s2.model())













