from pathlib import Path
import json,csv
root=Path(__file__).resolve().parent
base=json.loads((root/'comparison/baseline_results.json').read_text());rows=[]
for k,folder in [(3,root/'cycle_direct_k3'),(4,root/'cycle_recursive_k4')]:
 for r in base:
  if r['instance']=='arbiter' and r['K']==k:rows.append(dict(K=k,seed=r['seed'],method=r['tool']+' '+r['mode'],lambda_value=None,C=r['hypercut'],F_net=r['back_best_order'],J=None))
 for r in json.loads((folder/'pooled.json').read_text()):rows.append(dict(K=k,seed=r['seed'],method='CMA cycle archive',lambda_value=r['lambda'],C=r['cut'],F_net=r['feedback_net'],J=r['J']))
fields=list(rows[0]);out=root/'CYCLE_RESULTS.csv'
with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
lines=['# 正式循環目標重跑結果','','目標：J=C+lambda F_net。C 為超網 cut；F_net 為最小移除超網權重，使商圖無環。每次 CMA 評估都精確最小化全部區塊順序；不是固定區號 Back。','','設定：arbiter 同一訊號超圖，K3 direct / K4 joint recursive，seeds=42/43/44；每個方法、lambda 最多兩個 restart ×640 CMA evaluations，population16，sigma=.08/.25。lambda=0,.5,1,2,5,10；embedding seed42固定，容量上限1.02W/K。','','KaHyPar、dagP baseline 使用前輪同一輸入、相同 epsilon 與 seed 的已保存 pi，F_net 已精確重算；不冒充本輪重新執行 baseline。dagP + METIS 是嚴格無環端點，原生 edge-cut 目標不同但保留在主表。','','## Seed 42 主表','','| K | 方法 | lambda | C | F_net | J |','|---:|---|---:|---:|---:|---:|']
for r in rows:
 if r['seed']==42:lines.append(f"| {r['K']} | {r['method']} | {r['lambda_value'] if r['lambda_value'] is not None else '—'} | {r['C']:g} | {r['F_net']:g} | {r['J'] if r['J'] is not None else '—'} |")
lines+=['','## 三個 seed：lambda=5 的同目標評分','','| K | 方法 | seed42 J | seed43 J | seed44 J |','|---:|---|---:|---:|---:|']
for k in [3,4]:
 for method in ['KaHyPar direct','KaHyPar recursive','dagP strict_acyclic','CMA cycle archive']:
  rs=[r for r in rows if r['K']==k and r['method']==method and (r['lambda_value']==5 or r['lambda_value'] is None)]
  vals=[r['C']+5*r['F_net'] for r in sorted(rs,key=lambda r:r['seed'])]
  lines.append(f"| {k} | {method} | "+' | '.join(f'{v:g}' for v in vals)+' |')
lines+=['','raw results.json 與事後 archive pooled.json 分開。archive 從同 seed 所有方法及 lambda 的全部可行候選重選，保留完整 pi/p。完整 3-seed C/F/J 見 CYCLE_RESULTS.csv。','', '單一電路、固定特徵維度、小搜尋預算，未匹配各工具時間預算，不作 SOTA 或統計普遍性主張。F_net 是循環修復成本，不是循環個數；最大 SCC 只作診斷，沒有加入 objective。','', '功能檢查：test_cycle.py 包含獨立枚舉核對、所有改名的不變性、無環但固定逆向的案例、整條 net 與展開 arc 成本不同的案例。verify_cycle.py 重建保存 pi，獨立重算 F_net/J/容量及 archive 最佳性。']
(root/'CYCLE_RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
