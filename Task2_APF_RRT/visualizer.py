import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import List, Optional
from environment import Environment, Q_START, Q_GOAL
from apf_rrt_planner import PlannerResult

class Visualizer:
    def __init__(self, env: Environment): self.env=env
    def _eef(self, q): return self.env.eef_position(np.array(q))
    def _xyz(self, path): return np.array([self._eef(q) for q in path])

    def plot_comparison(self, baseline: PlannerResult, smoothed=None, save_path=None):
        fig,axes=plt.subplots(1,2,figsize=(18,8),subplot_kw={"projection":"3d"})
        for ax,show_tree in zip(axes,[True,False]):
            if show_tree and baseline.tree_edges:
                segs=[[self._eef(a),self._eef(b)] for a,b in baseline.tree_edges]
                ax.add_collection3d(Line3DCollection(segs,colors=[(0.55,0.55,0.55,0.15)],linewidths=0.4))
                ax.plot([],[],[],color="gray",alpha=0.4,lw=0.8,label=f"Tree ({len(baseline.tree_edges)} edges)")
            if baseline.path:
                xyz=self._xyz(baseline.path)
                ax.plot(xyz[:,0],xyz[:,1],xyz[:,2],color="#185FA5",lw=2.5,
                        label=f"APF-RRT  len={baseline.path_length:.3f} rad")
            if not show_tree and smoothed and len(smoothed)>1:
                sxyz=self._xyz(smoothed)
                ax.plot(sxyz[:,0],sxyz[:,1],sxyz[:,2],color="#3B6D11",lw=3.0,ls="--",
                        label=f"PSO-smoothed  wpts={len(smoothed)}")
                ax.scatter(sxyz[1:-1,0],sxyz[1:-1,1],sxyz[1:-1,2],color="#639922",s=28,zorder=6)
            ax.scatter(*self._eef(Q_START),color="#3B6D11",s=160,marker="*",zorder=10,label="Start")
            ax.scatter(*self._eef(Q_GOAL), color="#A32D2D",s=160,marker="*",zorder=10,label="Goal")
            self._obs(ax)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.set_xlim(-0.2,0.9); ax.set_ylim(-0.6,0.6); ax.set_zlim(0.0,1.1)
            ax.view_init(elev=28,azim=-55); ax.legend(fontsize=8,loc="upper left")
        axes[0].set_title("Phase A — APF-RRT baseline",fontsize=12)
        axes[1].set_title("Phase B — PSO-smoothed path",fontsize=12)
        fig.suptitle("Hybrid APF-RRT vs PSO-Enhanced Path",fontsize=14)
        plt.tight_layout()
        if save_path: plt.savefig(save_path,dpi=150,bbox_inches="tight"); print(f"[Viz] Saved → {save_path}")
        else: plt.show()

    def _obs(self, ax):
        for obs in self.env.obstacles:
            x,y,z=obs.position; r=obs.radius
            if obs.shape=="sphere":
                u=np.linspace(0,2*np.pi,14); v=np.linspace(0,np.pi,9)
                ax.plot_surface(x+r*np.outer(np.cos(u),np.sin(v)),
                                y+r*np.outer(np.sin(u),np.sin(v)),
                                z+r*np.outer(np.ones_like(u),np.cos(v)),
                                color="#E24B4A",alpha=0.18,linewidth=0)
            else:
                t=np.linspace(0,2*np.pi,24); h2=obs.height/2
                for dz in [-h2,h2]: ax.plot(x+r*np.cos(t),y+r*np.sin(t),
                                             np.full_like(t,z+dz),color="#378ADD",alpha=0.35,lw=0.8)
