import unittest
import numpy as np
from run import decode,metrics,reference_metrics,features
class Correctness(unittest.TestCase):
 def test_metrics_and_ordered_decoders(self):
  rng=np.random.default_rng(7)
  n=31; nets=[np.array(sorted({u,*rng.choice(np.arange(u+1,n),size=min(4,n-u-1),replace=False)})) for u in range(n-1)]
  drivers=np.arange(n-1); ew=rng.integers(1,9,len(nets)).astype(float)
  for _ in range(40):
   pi=rng.integers(0,5,n)
   self.assertEqual(metrics(pi,nets,ew,drivers),reference_metrics(pi,nets,ew,drivers))
  Xs,Xo,rank=features(nets,n,3,42,drivers)
  for layout in ('direct','recursive'):
   for k in (2,3,4,5):
    blocks=1 if layout=='direct' else k-1
    p=rng.uniform(.01,1,blocks*Xo.shape[1]);pi=decode(Xo,p,np.ones(n),rank,k,layout)
    self.assertEqual(set(pi),set(range(k)))
    self.assertLessEqual(np.bincount(pi).max()-np.bincount(pi).min(),1)
    self.assertEqual(reference_metrics(pi,nets,ew,drivers)[1],0)
 def test_global_hypernet_counted_once(self):
  nets=[np.array([0,1,2,3])];ew=np.array([7.]);pi=np.arange(4)
  self.assertEqual(reference_metrics(pi,nets,ew,np.array([3])),(7.,7.))
if __name__=='__main__':unittest.main()
