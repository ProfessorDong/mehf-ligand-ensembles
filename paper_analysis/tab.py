import json, csv, sys, statistics as st, os
R=sys.argv[1]
CH=['bond_length','bond_angle','dihedral','planar_improper','internal_rmsf']
sets={
 'ISO noproj s2.5':'internal_calibration_v1/noproj_f{f}_s2.5',
 'ISO proj s2.5':'internal_calibration_v1/proj_f{f}_s2.5',
 'GS noproj s2.5':'aniso_prior_v3/proj_noproj_f{f}_s2.5',
 'GS proj s2.5':'aniso_prior_v3/proj_f{f}_s2.5',
 'LAP noproj s2.5':'laplacian_prior_v1/noproj_f{f}_s2.5',
 'LAP proj s2.5':'laplacian_prior_v1/spectrum_f{f}_s2.5',
 'SIG noproj s2.5':'sigma_star_v1/noproj_f{f}_s2.5',
 'SIG spec_noproj s2.5':'sigma_star_v1/spectrum_noproj_f{f}_s2.5',
 'SIG proj s2.5':'sigma_star_v1/spectrum_f{f}_s2.5',
}
def spread_rows(path):
  rows=list(csv.DictReader(open(path)))
  sp=[]
  for r in rows:
    v=[r.get(c+'_ratio','') for c in CH[:4]]
    if all(x not in ('',None) for x in v):
      v=list(map(float,v)); sp.append(max(v)-min(v))
  return rows,sp
for name,pat in sets.items():
  print('==',name)
  for f in (1,2,3):
    p=os.path.join(R,pat.format(f=f))
    if not os.path.exists(p+'.json'): print('  f%d missing'%f); continue
    d=json.load(open(p+'.json'))
    meds=[d[c]['ratio_median'] for c in CH]
    ns=[d[c]['n_examples'] for c in CH]
    rows,sp=spread_rows(p+'.per_example.csv')
    som=max(meds[:4])-min(meds[:4])
    print('  f%d s=%s proj=%s n=%s  bond %.3f ang %.3f tor %.3f oop %.3f rmsf %.3f | spread-of-medians %.3f | median-of-spreads %.3f (n=%d)'%(
      f,d.get('internal_scale'),d.get('angle_projection'),ns,*meds,som,st.median(sp) if sp else float('nan'),len(sp)))
