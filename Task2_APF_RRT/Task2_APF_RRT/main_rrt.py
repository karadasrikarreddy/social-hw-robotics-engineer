import argparse,numpy as np
from environment import Environment,Q_START,Q_GOAL
from robot_kinematics import RobotKinematics
from apf_rrt_planner import APFRRTPlanner,APFRRTConfig
from pso_smoother import PSOSmoother
from visualizer import Visualizer
from benchmark import run_benchmark

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--gui",action="store_true")
    p.add_argument("--benchmark",action="store_true")
    p.add_argument("--save-plot",type=str,default=None)
    p.add_argument("--seed",type=int,default=42)
    return p.parse_args()

def single_run(gui,save_plot):
    print("="*60+"\n  Hybrid APF-RRT + PSO Motion Planner\n"+"="*60)
    env=Environment(gui=gui); kin=RobotKinematics(env)
    planner=APFRRTPlanner(env,kin,APFRRTConfig())
    smoother=PSOSmoother(kin); viz=Visualizer(env)
    print("\n[Main] Phase A — APF-RRT baseline...")
    result=planner.plan(Q_START.copy(),Q_GOAL.copy())
    if not result.success:
        print("[Main] No path found.")
        env.disconnect(); return
    print(f"[Main] ✓  nodes={result.node_count}  len={result.path_length:.4f} rad  t={result.computation_time:.2f}s")
    print("\n[Main] Phase B — PSO smoothing...")
    smoothed=smoother.smooth(result.path)
    rl=result.path_length; sl=PSOSmoother.path_length(smoothed)
    rs=PSOSmoother.path_smoothness(result.path); ss=PSOSmoother.path_smoothness(smoothed)
    print(f"\n  {'Metric':<22}{'Baseline':>12}{'PSO':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Path length (rad)':<22}{rl:>12.4f}{sl:>12.4f}")
    print(f"  {'Smoothness':<22}{rs:>12.4f}{ss:>12.4f}")
    print(f"  {'Waypoints':<22}{len(result.path):>12d}{len(smoothed):>12d}")
    print("\n[Main] Generating 3-D comparison plot...")
    viz.plot_comparison(result,smoothed,save_path=save_plot)
    env.disconnect()

def main():
    args=parse_args(); np.random.seed(args.seed)
    run_benchmark(gui=args.gui) if args.benchmark else single_run(gui=args.gui,save_plot=args.save_plot)

if __name__=="__main__": main()
