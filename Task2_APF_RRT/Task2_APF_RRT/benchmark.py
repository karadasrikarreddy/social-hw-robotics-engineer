import csv,time,numpy as np
from environment import Environment,Q_START,Q_GOAL
from robot_kinematics import RobotKinematics,JOINT_LOWER,JOINT_UPPER
from apf_rrt_planner import APFRRTPlanner,APFRRTConfig
from pso_smoother import PSOSmoother

N_RUNS=20; CSV_PATH="results.csv"; PERT=0.05

def run_benchmark(gui=False):
    env=Environment(gui=gui); kin=RobotKinematics(env)
    planner=APFRRTPlanner(env,kin,APFRRTConfig()); smoother=PSOSmoother(kin)
    ra_rows=[]; rb_rows=[]
    print(f"\n{'='*62}\n  APF-RRT Benchmark ({N_RUNS} runs)\n{'='*62}")
    for run in range(N_RUNS):
        rng=np.random.default_rng(seed=run)
        qs=np.clip(Q_START+rng.uniform(-PERT,PERT,7),JOINT_LOWER,JOINT_UPPER)
        qg=np.clip(Q_GOAL +rng.uniform(-PERT,PERT,7),JOINT_LOWER,JOINT_UPPER)
        ra=planner.plan(qs.copy(),qg.copy())
        ra_rows.append({"run":run+1,"phase":"A_baseline","success":int(ra.success),
            "time_s":round(ra.computation_time,3),"path_len":round(ra.path_length,4),
            "nodes":ra.node_count,"smoothness":round(PSOSmoother.path_smoothness(ra.path),4) if ra.path else 0.0})
        t0=time.time(); rb=planner.plan(qs.copy(),qg.copy())
        sm=smoother.smooth(rb.path) if rb.success else []; bt=time.time()-t0
        rb_rows.append({"run":run+1,"phase":"B_enhanced","success":int(rb.success),
            "time_s":round(bt,3),"path_len":round(PSOSmoother.path_length(sm),4),
            "nodes":rb.node_count,"smoothness":round(PSOSmoother.path_smoothness(sm),4)})
        print(f"  Run {run+1:2d}  A:{'✓' if ra.success else '✗'} {ra.computation_time:.1f}s  B:{'✓' if rb.success else '✗'} {bt:.1f}s")
    def agg(rows):
        ok=[r for r in rows if r["success"]]; n=len(rows); nan=float("nan")
        return {"sr":sum(r["success"] for r in rows)/n,
                "t":np.mean([r["time_s"] for r in ok]) if ok else nan,
                "l":np.mean([r["path_len"] for r in ok]) if ok else nan,
                "nd":np.mean([r["nodes"] for r in rows]),
                "sm":np.mean([r["smoothness"] for r in ok]) if ok else nan}
    sa=agg(ra_rows); sb=agg(rb_rows)
    print(f"\n{'='*62}")
    print(f"  {'Metric':<26}{'Phase A':>16}{'Phase B':>16}")
    print(f"  {'-'*58}")
    for lbl,a,b in [("Success Rate",f"{sa['sr']:.1%}",f"{sb['sr']:.1%}"),
                     ("Mean Time (s)",f"{sa['t']:.2f}",f"{sb['t']:.2f}"),
                     ("Mean Path Len (rad)",f"{sa['l']:.4f}",f"{sb['l']:.4f}"),
                     ("Mean Node Count",f"{sa['nd']:.1f}",f"{sb['nd']:.1f}"),
                     ("Mean Smoothness",f"{sa['sm']:.4f}",f"{sb['sm']:.4f}")]:
        print(f"  {lbl:<26}{a:>16}{b:>16}")
    print(f"{'='*62}\n")
    all_rows=ra_rows+rb_rows
    with open(CSV_PATH,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=all_rows[0].keys()); w.writeheader(); w.writerows(all_rows)
    print(f"[Benchmark] Written → {CSV_PATH}")
    env.disconnect()

if __name__=="__main__": run_benchmark()
