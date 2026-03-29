"""
pso_smoother.py — Phase B: PSO path smoother.
Cost = w_len*PathLength + w_smo*Curvature + w_col*CollisionPenalty
"""
import numpy as np
from typing import List
from robot_kinematics import RobotKinematics, JOINT_LOWER, JOINT_UPPER

N_PARTICLES=20; MAX_ITER=100; OMEGA_S=0.9; OMEGA_E=0.4; C1=1.5; C2=1.5
W_LEN=1.0; W_SMO=2.0; W_COL=40.0; V_MAX_F=0.08

class PSOSmoother:
    def __init__(self, kin: RobotKinematics):
        self.kin = kin
        self.vm_base = V_MAX_F*(JOINT_UPPER-JOINT_LOWER)   # shape (7,)

    def smooth(self, raw_path: List) -> List:
        if len(raw_path) <= 2: return raw_path
        qs=np.array(raw_path[0]); qg=np.array(raw_path[-1])
        nf=len(raw_path)-2; nd=7; dim=nf*nd
        rf=np.array(raw_path[1:-1]).flatten()

        # tile (7,) → (dim,) to match particle matrix shape (N, dim)
        noise = np.tile(0.05*(JOINT_UPPER-JOINT_LOWER), nf)
        vm    = np.tile(self.vm_base, nf)
        lo    = np.tile(JOINT_LOWER, nf)
        hi    = np.tile(JOINT_UPPER, nf)

        pos = np.clip(np.tile(rf,(N_PARTICLES,1))
                      + np.random.uniform(-noise,noise,size=(N_PARTICLES,dim)), lo, hi)
        vel = np.random.uniform(-vm, vm, size=(N_PARTICLES,dim))
        pb=pos.copy()
        pc=np.array([self._cost(p,qs,qg,nf,nd) for p in pb])
        gi=int(np.argmin(pc)); gb=pb[gi].copy(); gc=pc[gi]

        for it in range(MAX_ITER):
            w=OMEGA_S-(OMEGA_S-OMEGA_E)*(it/MAX_ITER)
            r1=np.random.rand(N_PARTICLES,dim); r2=np.random.rand(N_PARTICLES,dim)
            vel=np.clip(w*vel+C1*r1*(pb-pos)+C2*r2*(gb-pos),-vm,vm)
            pos=np.clip(pos+vel,lo,hi)
            for i in range(N_PARTICLES):
                c=self._cost(pos[i],qs,qg,nf,nd)
                if c<pc[i]:
                    pc[i]=c; pb[i]=pos[i].copy()
                    if c<gc: gc=c; gb=pos[i].copy()

        wps=gb.reshape(nf,nd)
        sm=[qs]+list(wps)+[qg]
        print(f"[PSO] Done — cost={gc:.4f}  waypoints={len(sm)}")
        return sm

    def _cost(self, flat, qs, qg, nf, nd):
        wps=np.vstack([qs,flat.reshape(nf,nd),qg])
        Jl=float(np.sum(np.linalg.norm(np.diff(wps,axis=0),axis=1)))
        Js=sum(float(np.linalg.norm(wps[i+1]-2*wps[i]+wps[i-1]))
               for i in range(1,len(wps)-1))
        Jc=0.0
        for q in wps[1:-1]:
            eef=self.kin.env.eef_position(q)
            for obs in self.kin.env.obstacles:
                d=np.linalg.norm(eef-obs.position)
                if d<obs.radius+0.06: Jc+=(obs.radius+0.06-d)**2
        return W_LEN*Jl+W_SMO*Js+W_COL*Jc

    @staticmethod
    def path_length(p):
        return sum(np.linalg.norm(np.array(p[i+1])-np.array(p[i])) for i in range(len(p)-1))

    @staticmethod
    def path_smoothness(p):
        if len(p)<3: return 0.0
        return sum(float(np.linalg.norm(np.array(p[i+1])-2*np.array(p[i])+np.array(p[i-1])))
                   for i in range(1,len(p)-1))
