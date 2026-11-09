import unittest,itertools
import numpy as np
from cycle_metric import feedback_net,reference_feedback_net
class TestCycle(unittest.TestCase):
 def test_exact_against_independent_orders(self):
  rng=np.random.default_rng(91)
  for k in (2,3,4):
   for repeat in range(12):
    pi=np.r_[np.arange(k),rng.integers(k,size=9)];n=len(pi)
    nets=[np.sort(rng.choice(n,size=int(rng.integers(2,6)),replace=False)) for _ in range(12)]
    drivers=np.array([rng.choice(p) for p in nets]);ew=rng.integers(1,8,len(nets)).astype(float)
    pins=np.concatenate(nets);starts=np.r_[0,np.cumsum(list(map(len,nets)))[:-1]]
    f,_=feedback_net(pi,pins,starts,drivers,ew,k)
    self.assertEqual(f,reference_feedback_net(pi,nets,drivers,ew,k))
    for order in itertools.permutations(range(k)):
     self.assertEqual(feedback_net(np.array(order)[pi],pins,starts,drivers,ew,k)[0],f)
 def test_acyclic_reversed_labels(self):
  pi=np.array([2,0,1]);pins=np.array([0,1,1,2]);starts=np.array([0,2]);d=np.array([0,1]);w=np.ones(2)
  self.assertEqual(feedback_net(pi,pins,starts,d,w,3)[0],0)
 def test_net_is_not_arc_cost(self):
  pi=np.array([0,1,2]);pins=np.array([0,1,2,0,1,0,2]);starts=np.array([0,3,5]);d=np.array([0,1,2]);w=np.array([1.,5.,5.])
  self.assertEqual(feedback_net(pi,pins,starts,d,w,3)[0],1)
if __name__=='__main__':unittest.main()
