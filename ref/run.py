#!/usr/bin/env python3
"""Full-weight CMA-ES, direct or joint-tree decoder; no historical CSV reuse."""
import argparse, hashlib, json, sys, time, platform
from pathlib import Path
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
import cma

def read_hgr(path):
    lines=[s.strip() for s in path.read_text().splitlines() if s.strip() and not s.lstrip().startswith('%')]
    header=list(map(int,lines[0].split())); m,n=header[:2]; fmt=header[2] if len(header)>2 else 0
    if fmt not in (0,1,10,11) or len(header)>3 and header[3]!=1: raise ValueError('Unsupported HGR format')
    nets=[]; ew=[]
    for line in lines[1:m+1]:
        a=list(map(int,line.split())); ew.append(a.pop(0) if fmt%10 else 1)
        pins=np.array(sorted(set(v-1 for v in a)),int)
        if len(pins)==0 or pins.min()<0 or pins.max()>=n: raise ValueError('Invalid pins')
        nets.append(pins)
    vw=np.array([int(s) for s in lines[m+1:]],float) if fmt//10 else np.ones(n)
    if len(nets)!=m or len(vw)!=n or np.any(vw<=0) or min(ew)<=0: raise ValueError('Invalid HGR weights/counts')
    return nets,np.array(ew,float),vw

def reference_metrics(pi,nets,ew,drivers=None):
    cut=back=0.
    for e,pins in enumerate(nets):
        labels=pi[pins]
        if labels.min()!=labels.max(): cut+=ew[e]
        if drivers is not None and np.any(labels<pi[drivers[e]]): back+=ew[e]
    return cut,back

_METRIC_CACHE={}
def metrics(pi,nets,ew,drivers=None):
    key=id(nets)
    if key not in _METRIC_CACHE:
        _METRIC_CACHE[key]=(np.concatenate(nets),np.r_[0,np.cumsum([len(p) for p in nets])[:-1]])
    pins,starts=_METRIC_CACHE[key]; labels=pi[pins]
    lo=np.minimum.reduceat(labels,starts);hi=np.maximum.reduceat(labels,starts)
    return float(ew[lo!=hi].sum()),float(ew[lo<pi[drivers]].sum()) if drivers is not None else 0.

def features(nets,n,dim,seed,drivers):
    rows=np.concatenate(nets); cols=np.repeat(np.arange(len(nets)),[len(p) for p in nets])
    H=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,len(nets)))
    dv=np.asarray(H.sum(1)).ravel(); inv=1/np.sqrt(np.maximum(dv,1)); de=1/np.array([len(p) for p in nets])
    def mv(x): return inv*(H@(de*(H.T@(inv*x))))
    vals,F=eigsh(LinearOperator((n,n),matvec=mv,dtype=float),k=min(dim+1,n-1),which='LA',v0=np.random.default_rng(seed).normal(size=n),tol=1e-7)
    F=F[:,np.argsort(-vals)[1:]]; F=(F-F.min(0))/np.maximum(np.ptp(F,axis=0),1e-12)
    Xs=F/np.maximum(F.std(0),1e-12)
    if drivers is None: return Xs,None,np.arange(n)
    out=[set() for _ in range(n)]; indeg=np.zeros(n,int)
    for pins,u in zip(nets,drivers):
        if u not in pins: raise ValueError('Driver must be a pin')
        for v in pins:
            if v!=u and v not in out[u]: out[u].add(int(v)); indeg[v]+=1
    topo=list(np.flatnonzero(indeg==0))
    for u in topo:
        for v in sorted(out[u]):
            indeg[v]-=1
            if indeg[v]==0: topo.append(v)
    if len(topo)!=n: raise ValueError('Directed input is cyclic; no feedback arcs removed')
    U=F.copy(); L=F.copy(); delta=1e-4*np.maximum(F.std(0),1e-12)
    for u in topo:
        for v in out[u]: U[v]=np.maximum(U[v],U[u]+delta)
    for u in reversed(topo):
        for v in out[u]: L[u]=np.minimum(L[u],L[v]-delta)
    Xo=np.concatenate((U,L),axis=1); Xo/=np.maximum(Xo.std(0),1e-12)
    rank=np.empty(n,int);rank[topo]=np.arange(n)
    return Xs,Xo,rank

def decode(X,p,weights,rank,k,layout):
    pi=np.empty(len(weights),int); dim=X.shape[1]
    def split(ids,lo,parts,node):
        if parts==1: pi[ids]=lo; return
        q=p[node*dim:(node+1)*dim]; order=ids[np.lexsort((rank[ids],X[ids]@q))]
        if layout=='direct':
            cum=np.cumsum(weights[order]); bounds=[0]
            for j in range(1,k):
                target=cum[-1]*j/k
                choices=np.arange(bounds[-1]+1,len(order)-(k-j)+1)
                boundary=int(choices[np.argmin(abs(cum[choices-1]-target))]); bounds.append(boundary)
            bounds.append(len(order))
            for j in range(k): pi[order[bounds[j]:bounds[j+1]]]=j
        else:
            left=parts//2; cum=np.cumsum(weights[order]); choices=np.arange(left,len(ids)-(parts-left)+1)
            b=int(choices[np.argmin(abs(cum[choices-1]-cum[-1]*left/parts))])
            # Preorder indexing gives exactly k-1 weight vectors, even for non-powers of two.
            split(order[:b],lo,left,node+1)
            split(order[b:],lo+left,parts-left,node+left)
    split(np.arange(len(weights)),0,k,0)
    return pi

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--hgr',type=Path,required=True);ap.add_argument('--drivers',type=Path)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--k',type=int,default=3);ap.add_argument('--dim',type=int,default=8)
    ap.add_argument('--lambdas',type=float,nargs='+',default=[0]);ap.add_argument('--seeds',type=int,nargs='+',default=[42])
    ap.add_argument('--evals',type=int,default=640);ap.add_argument('--popsize',type=int,default=16)
    ap.add_argument('--sigmas',type=float,nargs='+',default=[.08,.25]);ap.add_argument('--epsilon',type=float,default=.02)
    ap.add_argument('--layout',choices=['direct','recursive'],default='direct');a=ap.parse_args()
    if a.out.exists(): raise ValueError('Output must be new: refuse to overwrite results')
    nets,ew,vw=read_hgr(a.hgr);n=len(vw)
    if not 2<=a.k<=n or a.dim<1 or a.evals<a.popsize or min(a.sigmas)<=0 or a.epsilon<0 or min(a.lambdas)<0: raise ValueError('Invalid settings')
    drivers=np.array(json.loads(a.drivers.read_text()),int) if a.drivers else None
    if drivers is not None and (drivers.shape!=(len(nets),) or np.any(drivers<0) or np.any(drivers>=n)): raise ValueError('Invalid drivers')
    if drivers is None and any(a.lambdas): raise ValueError('HGR has no directions: nonzero lambda requires driver sidecar')
    a.out.mkdir(parents=True); (a.out/'run_source.py').write_bytes(Path(__file__).read_bytes()); t=time.perf_counter(); Xs,Xo,rank=features(nets,n,a.dim,a.seeds[0],drivers)
    np.savez_compressed(a.out/'embedding.npz',Xs=Xs,Xo=Xo if Xo is not None else np.empty((n,0)),rank=rank)
    modes={'spec':Xs} if Xo is None else {'order':Xo,'spec':Xs,'hybrid':np.concatenate((Xs,Xo),axis=1)}
    manifest={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'hgr_sha256':hashlib.sha256(a.hgr.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'cma':cma.__version__},'direction':'driver sidecar' if drivers is not None else 'absent; cut only','nodes':n,'nets':len(nets),'capacity_upper':(1+a.epsilon)*vw.sum()/a.k,'embedding_seconds':time.perf_counter()-t}
    if a.drivers: manifest['drivers_sha256']=hashlib.sha256(a.drivers.read_bytes()).hexdigest()
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[]; pure={}
    for seed in a.seeds:
      for mode,X in modes.items():
       d=X.shape[1]; blocks=1 if a.layout=='direct' else a.k-1
       for lam in a.lambdas:
        run=a.out/f'{mode}_seed{seed}_lam{lam:g}';run.mkdir();best=None;calls=0
        start=time.perf_counter(); trace=(run/'trace.jsonl').open('w'); generations=(run/'generations.jsonl').open('w')
        def evaluate(z,origin):
            nonlocal best,calls
            calls+=1; p=np.array(z).reshape(blocks,d).copy()
            norms=np.linalg.norm(p,axis=1,keepdims=True);p/=np.maximum(norms,1e-15)
            pi=decode(X,p.ravel(),vw,rank,a.k,a.layout);sizes=np.bincount(pi,weights=vw,minlength=a.k)
            valid=bool(np.all(sizes>0) and max(sizes)<=manifest['capacity_upper']+1e-9)
            cut,back=metrics(pi,nets,ew,drivers);J=cut+lam*back
            value=J if valid else (ew.sum()*(1+lam)+1)*(1+max(0,max(sizes)/manifest['capacity_upper']-1))
            row={'eval':calls,'origin':origin,'cut':cut,'back':back if drivers is not None else None,'J':J,'feasible':valid,'fitness':value}
            trace.write(json.dumps({**row,'p':p.ravel().tolist()})+'\n')
            if valid and (best is None or (J,back)<(best['J'],best['back_internal'])):
                best={**row,'back_internal':back,'part_weights':sizes.tolist()}
                np.savez_compressed(run/'best.npz',p=p.ravel(),pi=pi)
                np.savetxt(run/'best.part',pi,fmt='%d')
                (run/'best.json').write_text(json.dumps(best,indent=2))
            return value
        rng=np.random.default_rng(seed); anchors=[]
        if mode=='hybrid':
            for source in ('order','spec'):
                old=pure[(seed,source,lam)].reshape(blocks,-1);z=np.zeros((blocks,d))
                if source=='spec':z[:,:Xs.shape[1]]=old
                else:z[:,Xs.shape[1]:]=old
                anchors.append(z.ravel())
        else: anchors=[(np.ones((blocks,d)) if mode=='order' else rng.normal(size=(blocks,d))).ravel()]
        for z in anchors: evaluate(z,'anchor')
        for restart,sigma in enumerate(a.sigmas):
            mean=anchors[restart%len(anchors)].copy().reshape(blocks,d);mean/=np.maximum(np.linalg.norm(mean,axis=1,keepdims=True),1e-15)
            options={'seed':seed+restart*1009,'popsize':a.popsize,'maxfevals':a.evals,'verbose':-9,'verb_log':0,'verb_disp':0}
            if mode=='order': options['bounds']=[0,None];mean=np.maximum(mean,1e-8)
            es=cma.CMAEvolutionStrategy(mean.ravel(),sigma,options)
            while not es.stop() and es.countevals+a.popsize<=a.evals:
                zs=es.ask();fits=[evaluate(z,f'cma_restart{restart}') for z in zs];es.tell(zs,fits)
                generations.write(json.dumps({'restart':restart,'generation':es.countiter,'cma_evals':es.countevals,'sigma':float(es.sigma),'mean':es.mean.tolist(),'cov_diag':np.diag(es.C).tolist()})+'\n')
            generations.write(json.dumps({'restart':restart,'stop':str(es.stop()),'budget_exhausted':es.countevals+a.popsize>a.evals,'actual_cma_evals':es.countevals})+'\n')
        trace.close();generations.close()
        if best is None: raise RuntimeError('No capacity-feasible candidate; weighted decoder may need a stronger balancing algorithm')
        saved=np.load(run/'best.npz');c,b=reference_metrics(saved['pi'],nets,ew,drivers)
        assert c==best['cut'] and b==best['back_internal'] and c+lam*b==best['J']
        if mode!='hybrid':pure[(seed,mode,lam)]=saved['p'].copy()
        record={'mode':mode,'seed':seed,'lambda':lam,'layout':a.layout,'dimension':d*blocks,'evaluations_including_anchors':calls,'seconds':time.perf_counter()-start,'path':str(run),**best}
        records.append(record);(a.out/'results.json').write_text(json.dumps(records,indent=2));print(mode,seed,lam,c,b,'J',best['J'],flush=True)
    # Re-score every feasible evaluated direction; retain the complete selected solution.
    pooled=[]
    for seed in a.seeds:
      for lam in a.lambdas:
        winner=None
        for record in (r for r in records if r['seed']==seed):
          for line in (Path(record['path'])/'trace.jsonl').read_text().splitlines():
            cand=json.loads(line)
            if not cand['feasible']: continue
            back=cand['back'] if cand['back'] is not None else 0.
            key=(cand['cut']+lam*back,back)
            if winner is None or key<winner[0]: winner=(key,record,cand)
        key,record,cand=winner; X=modes[record['mode']]
        pi=decode(X,np.array(cand['p']),vw,rank,a.k,a.layout)
        destination=a.out/f'archive_seed{seed}_lam{lam:g}.npz'
        np.savez_compressed(destination,p=np.array(cand['p']),pi=pi)
        pooled.append({'seed':seed,'lambda':lam,'source':record['path'],'source_eval':cand['eval'],'mode':record['mode'],'saved_solution':str(destination),'cut':cand['cut'],'back':cand['back'],'J':key[0],'scope':'posthoc selection over ALL feasible evaluated directions'})
    (a.out/'pooled.json').write_text(json.dumps(pooled,indent=2))
if __name__=='__main__': main()
