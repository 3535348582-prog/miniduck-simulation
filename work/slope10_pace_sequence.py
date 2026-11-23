"""Search multi-step recovery sequences in native MuJoCo and test closed loop."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from slope10_downhill_continuation_20261123 import (
    A, BASE, CONTRACT, CONTEXT, DT, MissionConfig, RebuiltHeadingPolicy,
    STATE_SPEC, init, mujoco, np, p070_command, sha, sources,
)

BASES=np.array([
    [0,0,1,0,0,0,0,-1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,0,1,0,0,0,0,1,0],
    [0,0,0,1,0,0,0,0,-1,0],
    [0,0,0,0,1,0,0,0,0,1],
    [0,0,0,0,1,0,0,0,0,-1],
    [0,1,0,0,0,0,1,0,0,0],
    [0,1,0,0,0,0,-1,0,0,0],
],dtype=np.float32)
STATE_FIELDS=("actions","targets","phase","mode","heading","reference_heading")
BLOCK=6
PLAN_BLOCKS=7
CAP=.55


def factor(controls):
    return np.clip(controls@BASES,-CAP,CAP)


def observe(policy, config):
    command=p070_command(A.data.qpos[0],A.data.qpos[1],config)
    return policy.build(A.sensor("gyro"),A.sensor("accelerometer"),
                        [command.vx,command.vy,command.yaw_rate],A.data.qpos[A.qaddr],
                        A.data.qvel[A.daddr],A.contacts(),heading=A.heading())


def physics_step(target):
    if np.max(np.abs(A.data.ctrl-target))>5.24*DT+1e-6:
        raise ValueError("Target exceeds joint step bound")
    A.data.ctrl[:]=target
    mujoco.mj_step(A.model,A.data,nstep=10)
    mujoco.mj_forward(A.model,A.data)
    r=A.data.site_xmat[A.imu].reshape(3,3)
    tilt=float(np.degrees(np.arccos(np.clip(r[2,2],-1.,1.))))
    failed=bool(A.data.qpos[2]<.05 or tilt>75. or A.wall_hit() or A.body_ground_hit())
    return r,tilt,failed


def step(actor, action):
    target=actor.target(action)
    actor.commit(action,target)
    return physics_step(target)


class SequencePlanner:
    def __init__(self, actor, cfg, seed, population, iterations, late_balance=False,
                 guard_lateral=False, terminal_aware=False, staged_guard=False, descent_safe=False,
                 full_stop_preview=False):
        self.actor=actor
        self.forward=actor.infer
        self.template=copy.copy(actor)
        self.cfg=cfg
        self.rng=np.random.default_rng(seed)
        self.population=population
        self.iterations=iterations
        self.late_balance=late_balance
        self.guard_lateral=guard_lateral
        self.terminal_aware=terminal_aware
        self.staged_guard=staged_guard
        self.descent_safe=descent_safe
        self.slope_weight=0.
        self.full_stop_preview=full_stop_preview
        self.guard_latched=not staged_guard
        self.guard_latched_x=None
        self.mean=np.zeros((PLAN_BLOCKS,len(BASES)),np.float32)
        self.block_left=0
        self.option=np.zeros(10,np.float32)
        self.records=[]

    def score(self, integration, previous, x0, control):
        data=mujoco.MjData(A.model)
        mujoco.mj_setState(A.model,data,integration,STATE_SPEC)
        mujoco.mj_forward(A.model,data)
        branch=copy.copy(self.template)
        for key in STATE_FIELDS:
            setattr(branch,key,copy.copy(getattr(self.actor,key)))
        risk=0.
        route_risk=0.
        momentum_risk=0.
        slope_motion_risk=0.
        balance=self.late_balance and (
            x0>=3.32 or (self.guard_lateral and self.guard_latched and
                         x0>=1.5 and abs(float(previous.qpos[1]))>.105))
        near_finish=self.terminal_aware and x0>=4.15
        calm_descent=self.descent_safe and 2.48<=x0<4.30
        brake_started=None
        brake_from=None
        residuals=factor(control)
        try:
            A.data=data
            for phase,residual in enumerate(residuals):
                for tick in range(BLOCK):
                    if near_finish and (brake_started is not None or data.qpos[0]>=self.cfg.stop_x):
                        if brake_started is None:
                            brake_started=float(data.time)
                            brake_from=np.asarray(data.ctrl,dtype=np.float32).copy()
                        blend=min(1.,(float(data.time)-brake_started)/.45)
                        desired=np.clip(brake_from+blend*(branch.default-brake_from),
                                        branch.limits[:,0],branch.limits[:,1])
                        limit=branch.contract["max_motor_velocity"]*DT
                        target=np.clip(desired,data.ctrl-limit,data.ctrl+limit)
                        r,tilt,failed=physics_step(target)
                        branch.targets=np.asarray(target,dtype=np.float32)
                    else:
                        command=observe(branch,self.cfg)
                        action=np.clip(branch.infer(command)+residual,-1.,1.)
                        r,tilt,failed=step(branch,action)
                    if failed:
                        return -20.+(phase*BLOCK+tick)/40.,True
                    pitch=float(np.degrees(np.arcsin(np.clip(-r[2,0],-1.,1.))))
                    roll=float(np.degrees(np.arctan2(r[2,1],r[2,2])))
                    risk+=(max(tilt-42.,0.)/25.)**2+(max(abs(roll)-14.,0.)/26.)**2
                    risk+=.6*(max(pitch-37.,-pitch-6.,0.)/30.)**2
                    risk+=.3*(max(float(data.qvel[0])-.23,0.)/.19)**2
                    if self.slope_weight and .80 <= data.qpos[0] < 4.20:
                        angular = float(np.linalg.norm(data.qvel[3:5]))
                        slope_motion_risk += (max(tilt - 40., 0.) / 18.) ** 2
                        slope_motion_risk += .40 * (max(angular - 2.6, 0.) / 4.) ** 2
                        slope_motion_risk += .75 * (not np.any(A.contacts()))
                    if calm_descent and data.qpos[0]<4.26:
                        forward=float(data.qvel[0])
                        horizontal=float(np.linalg.norm(data.qvel[:2]))
                        momentum_risk+=(max(forward-.23,0.)/.22)**2
                        momentum_risk+=.65*(max(horizontal-.37,0.)/.28)**2
                        angular=float(np.linalg.norm(data.qvel[3:5]))
                        momentum_risk+=.25*(max(angular-3.2,0.)/5.)**2
                    if balance:
                        y=float(data.qpos[1])
                        lateral=float(data.qvel[1])
                        heading=A.heading()
                        route_risk+=(max(abs(y)-.07,0.)/.16)**2
                        route_risk+=.5*(max(abs(lateral)-.12,0.)/.28)**2
                        route_risk+=.5*(max(abs(heading)-.17,0.)/.55)**2
                        route_risk+=.3*(max(float(data.qvel[0])-.26,0.)/.25)**2
            dx=float(data.qpos[0])-x0
            penalty=1.4*risk/(PLAN_BLOCKS*BLOCK)+.45*(max(tilt-36.,0.)/25.)**2
            penalty += self.slope_weight * slope_motion_risk / (PLAN_BLOCKS * BLOCK)
            if balance:
                penalty+=2.4*route_risk/(PLAN_BLOCKS*BLOCK)
                penalty+=1.2*(max(abs(float(data.qpos[1]))-.06,0.)/.16)**2
                penalty+=.40*(max(abs(float(data.qvel[1]))-.10,0.)/.25)**2
            if brake_started is not None:
                distance=float(np.linalg.norm(data.qpos[:2]-self.cfg.goal_xy))
                penalty+=6.*max(distance-.08,0.)+.35*float(np.linalg.norm(data.qvel[:2]))
                if self.full_stop_preview and x0>=4.50:
                    settled=0
                    restored=False
                    limit=branch.contract["max_motor_velocity"]*DT
                    for tick in range(141):
                        blend=min(1.,(float(data.time)-brake_started)/.45)
                        desired=np.clip(brake_from+blend*(branch.default-brake_from),
                                        branch.limits[:,0],branch.limits[:,1])
                        target=np.clip(desired,data.ctrl-limit,data.ctrl+limit)
                        _,_,failed=physics_step(target)
                        if failed:
                            return -14.+tick/145.,True
                        distance=float(np.linalg.norm(data.qpos[:2]-self.cfg.goal_xy))
                        speed=float(np.linalg.norm(data.qvel[:2]))
                        angular=float(np.linalg.norm(data.qvel[3:6]))
                        stable=distance<=.10 and speed<.02 and abs(data.qvel[5])<.10 and angular<.20
                        settled=settled+1 if stable else 0
                        if settled>=round(self.cfg.settle_s/DT):
                            restored=True
                            break
                    penalty+=9.*max(distance-.075,0.)
                    penalty+=-1.5 if restored else .85
            if calm_descent:
                penalty+=1.8*momentum_risk/(PLAN_BLOCKS*BLOCK)
                penalty+=7.*max(dx-.15,0.)
            energy=.10*float(np.mean(residuals**2))
            return 14.*dx-3.*abs(float(data.qpos[1]))-penalty-energy,False
        finally:
            A.data=previous

    def infer(self, obs):
        raw=self.forward(obs)
        if not self.block_left:
            old=A.data
            x=float(old.qpos[0])
            if self.staged_guard and not self.guard_latched and 1.50<=x<2.32 and abs(float(old.qpos[1]))>.105:
                self.guard_latched=True
                self.guard_latched_x=x
            state=np.empty(mujoco.mj_stateSize(A.model,STATE_SPEC))
            mujoco.mj_getState(A.model,old,state,STATE_SPEC)
            mean=self.mean
            best=mean.copy()
            best_score=-np.inf
            best_failed=True
            std=np.full_like(mean,.23)
            for k in range(self.iterations):
                choices=self.rng.normal(mean,std,size=(self.population,*mean.shape)).astype(np.float32)
                choices=np.clip(choices,-CAP,CAP)
                choices[0]=best
                choices[1]=0.
                ranked=sorted(((self.score(state,old,x,s),i) for i,s in enumerate(choices)),
                              key=lambda item:item[0][0],reverse=True)
                (score, failed), i=ranked[0]
                if score>best_score:
                    best_score,best_failed=score,failed
                    best=choices[i].copy()
                elite=choices[[i for _,i in ranked[:max(3,self.population//5)]]]
                mean=.35*mean+.65*elite.mean(axis=0)
                std=np.maximum(.09,.50*std+.50*elite.std(axis=0))
            self.option=factor(best[0])
            self.mean=np.concatenate((best[1:],np.zeros_like(best[:1])),axis=0)
            self.block_left=BLOCK
            self.records.append(dict(x=x,time=float(old.time),score=float(best_score),
                                     would_fall=best_failed,max_action=float(np.max(np.abs(self.option)))))
        self.block_left-=1
        return np.clip(raw+self.option,-1.,1.)


def rollout(job):
    label, snapshot, use_planner, args, index=job
    A.data=mujoco.MjData(CONTEXT["model"])
    mujoco.mj_setState(A.model,A.data,snapshot["state"],STATE_SPEC)
    mujoco.mj_forward(A.model,A.data)
    policy=RebuiltHeadingPolicy.load(str(BASE),str(CONTRACT))
    for field in STATE_FIELDS:
        setattr(policy,field,copy.copy(snapshot[field]))
    policy.mode=tuple(int(i) for i in policy.mode)
    if np.max(np.abs(A.data.ctrl-policy.targets))>1e-6:
        raise ValueError("Restored actuator history is inconsistent")
    config=MissionConfig(command_mode="p070",terrain="slope10",goal_xy=(5.,0.),
                         contract=CONTEXT["contract"],trace=False,dance=False,stop_x=4.96)
    config.profile["uphill"],config.profile["exit"]=.70,.10
    begin=float(A.data.qpos[0])
    planner=SequencePlanner(policy,config,4731+index,args.population,args.iterations) if use_planner else None
    max_x=begin
    states=[]
    training=[]
    for tick in range(int(round(args.seconds/DT))):
        obs=observe(policy,config)
        action=planner.infer(obs) if planner is not None else policy.infer(obs)
        _,tilt,failed=step(policy,action)
        x=float(A.data.qpos[0])
        max_x=max(max_x,x)
        states.append((x,float(A.data.qpos[1]),tilt))
        if planner is not None:
            training.append((obs.copy(),action.copy()))
        if failed or x>=4.20:
            break
    controller="sequence" if use_planner else "accepted"
    artifact=args.out/f"{controller}_{index}.npz"
    data=dict(trace=np.asarray(states))
    if training:
        obs,actions=zip(*training)
        data.update(observations=np.asarray(obs),actions=np.asarray(actions))
    np.savez_compressed(artifact,**data)
    result=dict(mode=controller,seed_source=label,index=index,x0=begin,x_max=max_x,
                fell=failed,segment_exit=bool(x>=4.20 and not failed),duration=(tick+1)*DT,
                record=artifact.name,record_sha256=sha(artifact),
                decisions=planner.records if planner is not None else [])
    print(json.dumps({k:v for k,v in result.items() if k!="decisions"}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--snapshots",type=Path,nargs="+",required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--seconds",type=float,default=7.)
    parser.add_argument("--population",type=int,default=20)
    parser.add_argument("--iterations",type=int,default=2)
    args=parser.parse_args()
    assert mujoco.__version__=="3.6.0"
    args.out.mkdir()
    starts=[(label,s) for label,s in sources(args.snapshots)
            if 2.45<=float(s["x"])<=2.82 and float(s["threshold_x"]) in (2.46,2.68,2.80)]
    jobs=[(label,s,active,args,i)
          for i,(label,s) in enumerate(starts) for active in (False,True)]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init) as pool:
        rows=list(pool.map(rollout,jobs))
    report=dict(terrain="slope10",scope="Native continuation only",
                model_sha256=sha(BASE),envelope=CAP,planner_horizon=PLAN_BLOCKS*BLOCK,
                rows=rows)
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
