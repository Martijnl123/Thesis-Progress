# Draft 7b (27 Nov 2026): Figures 2(b) and 5(a) without the eight Magenta Security positions (post hoc, App. C (o)).
# Runs draft6/d6_figs.py unchanged with the same filter as d7b_step2_nomagenta.py in d2_step2.build(); the other
# figures come out identical. Rewritten into the repository on 27 Nov 2026 from the description in the draft 7b build log
# (the wrapper used that morning was not saved); its output matches the draft 7b to 7k figures pixel for pixel.
# Usage: python3 d7b_figs_nomagenta.py <draft folder>   (needs numbers-sep.tex in that folder, as d6_figs.py does)
import os, sys, runpy
H=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(H,'..','draft2')); sys.path.insert(0,os.path.join(H,'..'))
import d2_step2 as S2
MAGENTA={41678,41679,41680,43748,43749,43751,43752,51490}
_build=S2.build
def build_nm(p,matchfile,labmode='model'):
    q=_build(p,matchfile,labmode); return q[~q.pkey.isin(MAGENTA)].copy()
S2.build=build_nm
sys.argv=[os.path.join(H,'..','draft6','d6_figs.py'),sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
