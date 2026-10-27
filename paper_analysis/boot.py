import csv, sys, os, random, statistics as st
R=sys.argv[1]; O=sys.argv[2]
CH=['bond_length','bond_angle','dihedral','planar_improper']
arms={'ISO':'internal_calibration_v1/noproj_f{f}_s2.5','GS':'aniso_prior_v3/proj_noproj_f{f}_s2.5',
      'LAP':'laplacian_prior_v1/noproj_f{f}_s2.5','SIG':'sigma_star_v1/noproj_f{f}_s2.5'}
uni={}
for f in (1,2,3):
  for r in csv.DictReader(open(f'{O}/sigma_star_v1/full_f{f}/fit_examples.csv')):
    uni[(f,r['parent']+'->'+r['product'])]=r['uniprot']
data={}
for a,p in arms.items():
  for f in (1,2,3):
    for r in csv.DictReader(open(f'{R}/{p.format(f=f)}.per_example.csv')):
      data[(a,f,r['key'])]=r
keys={f:[k for (a,ff,k) in data if a=='SIG' and ff==f] for f in (1,2,3)}
# pairing check
for f in (1,2,3):
  s=[set(k for (a,ff,k) in data if a==A and ff==f) for A in arms]
  print('fold',f,'identical key sets across arms:',all(x==s[0] for x in s),'n=',len(s[0]),'targets=',len({uni[(f,k)] for k in s[0]}))
def val(a,f,k,metric):
  r=data[(a,f,k)]
  if metric=='spread':
    v=[r[c+'_ratio'] for c in CH]
    if any(x=='' for x in v): return None
    v=list(map(float,v)); return max(v)-min(v)
  x=r[metric+'_ratio']; return float(x) if x!='' else None
random.seed(20261013)
for metric in ('planar_improper','spread','bond_angle'):
 for comp in ('ISO','LAP','GS'):
  per=[]
  for f in (1,2,3):
    for k in keys[f]:
      s,c=val('SIG',f,k,metric),val(comp,f,k,metric)
      if s is not None and c is not None: per.append((f,uni[(f,k)],s-c,s<c))
  # cluster = (fold, uniprot): folds have disjoint training sets; within a fold cluster by target
  cl={}
  for f,u,d,w in per: cl.setdefault((f,u),[]).append(d)
  ids=list(cl); B=10000; meds=[]
  for _ in range(B):
    smp=[d for i in random.choices(ids,k=len(ids)) for d in cl[i]]
    meds.append(st.median(smp))
  meds.sort()
  diffs=[d for *_,d,_ in per]
  print(f'{metric:16s} SIG-{comp}: n={len(per)} clusters={len(ids)} median paired diff {st.median(diffs):+.3f} 95% CI [{meds[249]:+.3f}, {meds[9749]:+.3f}]  SIG lower {100*sum(w for *_,w in per)/len(per):.1f}%')
