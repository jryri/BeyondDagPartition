"""Reconstruct saved partitions from saved features/weights and audit full-graph metrics."""
import json
from pathlib import Path
from collections import Counter
from itertools import permutations
import numpy as np
from run import read_hgr,decode,reference_metrics

def verify(root):
 m=json.loads((root/'manifest.json').read_text());a=m['args'];nets,ew,vw=read_hgr(Path(a['hgr']))
 drivers=np.array(json.loads(Path(a['drivers']).read_text())) if a['drivers'] else None
 embed=np.load(root/'embedding.npz'); rows=[]
 for r in json.loads((root/'results.json').read_text()):
  path=Path(r['path']);z=np.load(path/'best.npz');pi=z['pi'];mode=r['mode']
  X=embed['Xs'] if mode=='spec' else embed['Xo'] if mode=='order' else np.concatenate((embed['Xs'],embed['Xo']),axis=1)
  assert np.array_equal(decode(X,z['p'],vw,embed['rank'],a['k'],a['layout']),pi)
  assert np.array_equal(np.loadtxt(path/'best.part',dtype=int),pi)
  cut,back=reference_metrics(pi,nets,ew,drivers)
  assert cut==r['cut'] and back==r['back_internal'] and cut+r['lambda']*back==r['J']
  sizes=np.bincount(pi,weights=vw,minlength=a['k']); assert np.all(sizes>0) and max(sizes)<=m['capacity_upper']+1e-9
  trace=[json.loads(s) for s in (path/'trace.jsonl').read_text().splitlines()]
  assert len(trace)==r['evaluations_including_anchors']
  assert r['J']==min(t['J'] for t in trace if t['feasible'])
  assert all(abs(t['fitness']-t['J'])<1e-8 for t in trace if t['feasible'])
  row={'mode':mode,'lambda':r['lambda'],'cut':cut,'back':back if drivers is not None else None,'relative_upper_imbalance':a['k']*max(sizes)/sum(vw)-1,'verified':True}
  if drivers is not None:
   k=a['k'];Q=Counter()
   for e,pins in enumerate(nets):
    u=drivers[e]
    for v in pins:
     if pi[u]!=pi[v]:Q[(int(pi[u]),int(pi[v]))]+=float(ew[e])
   reach=np.eye(k,dtype=bool)
   for u,v in Q:reach[u,v]=True
   for t in range(k):reach|=reach[:,t,None]&reach[t,None,:]
   row['max_scc_size']=max(int(np.sum(reach[u]&reach[:,u])) for u in range(k))
   costs=[]
   for order in permutations(range(k)):
    rank={v:i for i,v in enumerate(order)}; costs.append(sum(w for (u,v),w in Q.items() if rank[u]>rank[v]))
   row['minimum_feedback_arc_cost_expanded_dependencies']=min(costs)
  rows.append(row)
 for item in json.loads((root/'pooled.json').read_text()):
  if 'saved_solution' not in item: continue
  z=np.load(item['saved_solution']);c,b=reference_metrics(z['pi'],nets,ew,drivers)
  assert c==item['cut'] and (b==item['back'] if drivers is not None else item['back'] is None)
  assert c+item['lambda']*b==item['J']
  X=embed['Xs'] if item['mode']=='spec' else embed['Xo'] if item['mode']=='order' else np.concatenate((embed['Xs'],embed['Xo']),axis=1)
  assert np.array_equal(decode(X,z['p'],vw,embed['rank'],a['k'],a['layout']),z['pi'])
  candidates=[]
  for r in json.loads((root/'results.json').read_text()):
   if r['seed']!=item['seed']:continue
   for line in (Path(r['path'])/'trace.jsonl').read_text().splitlines():
    t=json.loads(line)
    if t['feasible']:candidates.append(t['cut']+item['lambda']*(t['back'] or 0))
  assert item['J']==min(candidates)
 (root/'verification.json').write_text(json.dumps(rows,indent=2))
 return rows
if __name__=='__main__':
 import sys
 for arg in sys.argv[1:]: print(arg,json.dumps(verify(Path(arg))))
