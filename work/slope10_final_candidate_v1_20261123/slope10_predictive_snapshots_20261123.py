"""Predictive native simulation teacher for the 10-degree uphill entry."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import mujoco
import numpy as np

ROOT=Path(os.environ["MINIDUCK_ROOT"])
TRAIN=Path(os.environ["MINIDUCK_TRAINING"])
sys.path[:0]=[str(ROOT),str(TRAIN/"training"),str(TRAIN/"vendor")]
import acceptance_5m as A
import control.mission_runner as mission
from control.mission_fsm import MotionCommand
from control.mission_runner import MissionConfig,load_contract,p070_command as original_p070,run_mission
from perception.goal_adapter import CoordinateGoalAdapter,GoalSpec
from rebuilt_policy import RebuiltHeadingPolicy

BASE=ROOT/"models/duck_initial/slope6_ppo_p3_128k.npz"
CONTRACT=ROOT/"models/duck_initial/accepted_6deg_final.json"
STATE_SPEC=mujoco.mjtState.mjSTATE_INTEGRATION


def p070_command(x,y,config):
    command=original_p070(x,y,config)
    target=config.profile.get("downhill_transition_vx")
    if target is None or not 2.48<=x<4.20:
        return command
    weight=min(1.,(x-2.48)/.24)
    vx=command.vx+weight*(target-command.vx)
    limit=min(.04,.45*max(vx,1e-6))
    vy=float(np.clip(config.goal_xy[1]-y,-limit,limit))
    return MotionCommand(vx,vy,0.,1.,"p070")


def choices(kinematic=False):
    patterns = [
        ("hips",(2,7)), ("knees",(3,8)), ("ankles",(4,9)),
        ("hip_knee",(2,3,7,8)), ("left_leg",(2,3,4)), ("right_leg",(7,8,9)),
        ("hip_ankle",(2,4,7,9)),
    ]
    values=[("none",np.zeros(10,dtype=np.float32))]
    sagittal_signs={2:1.,3:1.,4:1.,7:-1.,8:1.,9:1.}
    for name, indexes in patterns:
        for gain in (-.20,.20):
            vector=np.zeros(10,dtype=np.float32)
            for i in indexes:
                vector[i]=gain*(sagittal_signs[i] if kinematic else (-1 if i<5 else 1))
            values.append((f"{name}_{gain:+.2f}",vector))
    return values


def capture_risk(model, data, body, stance_spans):
    """Approximate dynamic support using projected COM and real sole contacts."""
    support = []
    for contact in data.contact:
        first, second = contact.geom
        if contact.dist <= 0 and (
            (first in A.feet and second in A.ground)
            or (second in A.feet and first in A.ground)
        ):
            support.append(contact.pos)
    if not support:
        return .12
    support = np.asarray(support)
    mujoco.mj_subtreeVel(model, data)
    com = data.subtree_com[body]
    height = float(np.clip(com[2] - support[:, 2].mean(), .12, .40))
    capture = com[:2] + np.sqrt(height / 9.81) * data.subtree_linvel[body, :2]
    lo = support[:, :2].min(axis=0) - stance_spans
    hi = support[:, :2].max(axis=0) + stance_spans
    outside = np.maximum(np.maximum(lo - capture, capture - hi), 0.)
    return float(np.sum((outside / [.09, .05]) ** 2))


def recovery_choices():
    for name, delta in choices(kinematic=True)[1:]:
        yield f"physical_{name}", delta
        yield f"strong_{name}", delta*1.75
    for gain in (-.30,.30):
        for name, slots, signs in (
            ("roll", (1,6), (1,1)),
            ("cross_roll", (1,6), (1,-1)),
            ("hip_yaw", (0,5), (1,1)),
        ):
            delta=np.zeros(10,np.float32)
            delta[list(slots)]=gain*np.asarray(signs)
            yield f"{name}_{gain:+.2f}",delta


class EntryPlanner:
    def __init__(self, policy, config, start, end, downhill_aware, horizon, crest_horizon,
                 crest_speed, crest_phase, preview_hold, dynamic_balance, kinematic_choices, recovery_bank):
        self.policy=policy
        self.template=copy.copy(policy)
        self.forward=policy.infer
        self.config=config
        self.start=start
        self.end=end
        self.downhill_aware=downhill_aware
        self.horizon=horizon
        self.crest_horizon=crest_horizon
        self.crest_speed=crest_speed
        self.crest_phase=crest_phase
        self.preview_hold=preview_hold
        self.dynamic_balance=dynamic_balance
        self.recovery=recovery_bank
        self.expanded=list(recovery_choices()) if recovery_bank else []
        self.phase_rate=1.
        self.remaining=0
        self.applied=np.zeros(10,np.float32)
        self.records=[]
        self.samples=[]
        self.options=choices(kinematic_choices)

    def commit(self, action, target):
        before=self.policy.phase
        self.base_commit(action,target)
        self.policy.phase=(before+self.phase_rate)%self.policy.contract["phase_period_steps"]

    def simulate(self, old, initial, perturbation, model, body, horizon, phase_rate=1., hold=6, follow=0.):
        data=mujoco.MjData(model)
        mujoco.mj_setState(model,data,initial,STATE_SPEC)
        mujoco.mj_forward(model,data)
        branch=copy.copy(self.template)
        for key in ("actions","targets","phase","mode","heading","reference_heading"):
            setattr(branch,key,copy.copy(getattr(self.policy,key)))
        trace=[]
        total_risk=0.
        balance_risk=0.
        spans=np.array([.035,.023])
        try:
            A.data=data
            for tick in range(horizon):
                command=p070_command(data.qpos[0],data.qpos[1],self.config)
                obs=branch.build(A.sensor("gyro"),A.sensor("accelerometer"),
                                 [command.vx,command.vy,command.yaw_rate],
                                 data.qpos[A.qaddr],data.qvel[A.daddr],A.contacts(),heading=A.heading())
                gate=self.start<=data.qpos[0]<self.end
                residual=(perturbation if tick<hold else (follow if tick<hold+8 else 0.)) if gate else 0.
                action=np.clip(branch.infer(obs)+residual,-1.,1.)
                target=branch.target(action)
                before=branch.phase
                branch.commit(action,target)
                if tick<6 and gate and phase_rate!=1.:
                    branch.phase=(before+phase_rate)%branch.contract["phase_period_steps"]
                data.ctrl[:]=target
                for _ in range(10):
                    mujoco.mj_step(model,data)
                mujoco.mj_forward(model,data)
                r=data.xmat[body].reshape(3,3)
                pitch=math.degrees(math.asin(float(np.clip(-r[2,0],-1,1))))
                imu_up=data.site_xmat[A.imu].reshape(3,3)[2,2]
                tilt=math.degrees(math.acos(float(np.clip(imu_up,-1,1))))
                trace.append((float(data.qpos[0]), pitch, tilt))
                total_risk+=(max(-pitch-5.,0.)/35.)**2+(max(tilt-43.,0.)/25.)**2
                if self.downhill_aware and 2.70 <= data.qpos[0] <= 4.22:
                    total_risk+=(max(pitch-40.,0.)/25.)**2
                if self.crest_speed is not None and 2.38 <= data.qpos[0] <= 3.05:
                    total_risk+=(max(float(data.qvel[0])-self.crest_speed,0.)/.13)**2
                if self.dynamic_balance and 1.85 <= data.qpos[0] < 4.22:
                    angular = np.linalg.norm(data.qvel[3:5])
                    balance_risk += (
                        capture_risk(model,data,body,spans)
                        + (max(tilt-36.,0.)/23.)**2
                        + .25*(max(float(angular)-2.5,0.)/4.)**2
                    )
                if data.qpos[2]<.05 or tilt>75. or A.wall_hit() or A.body_ground_hit():
                    return -1000.+len(trace), True, trace
        finally:
            A.data=old
        dx=trace[-1][0]-float(old.qpos[0])
        lateral=abs(float(data.qpos[1]-self.config.goal_xy[1]))
        score=12.*dx-2.*lateral-(1.4*total_risk+.85*balance_risk)/horizon
        return score,False,trace

    def infer(self,obs):
        raw=self.forward(obs)
        x=float(A.data.qpos[0])
        if not self.start<=x<self.end:
            self.remaining=0
            self.phase_rate=1.
            return raw
        if self.remaining==0:
            model,old=A.model,A.data
            initial=np.empty(mujoco.mj_stateSize(model,STATE_SPEC))
            mujoco.mj_getState(model,old,initial,STATE_SPEC)
            body=int(model.jnt_bodyid[model.joint("floating_base").id])
            ranked=[]
            horizon=self.crest_horizon if self.crest_horizon and 2.35 <= x <= 3.10 else self.horizon
            hold=self.preview_hold if 1.85 <= x < 4.22 else 6
            for key,delta in self.options:
                score,failed,trace=self.simulate(old,initial,delta,model,body,horizon,hold=hold)
                ranked.append((score,key,delta,failed,trace,1.))
            if self.crest_phase and 2.35 <= x < 3.24:
                candidates=sorted(ranked,key=lambda row:row[0],reverse=True)[:3]
                for _,key,delta,_,_,_ in candidates:
                    for rate in (.70,1.30):
                        score,failed,trace=self.simulate(old,initial,delta,model,body,horizon,rate,hold)
                        ranked.append((score,key,delta,failed,trace,rate))
            best=max(ranked,key=lambda row:row[0])
            if self.recovery and x>=1.85 and (best[3] or best[4][-1][2]>50.):
                for key,delta in self.expanded:
                    score,failed,trace=self.simulate(old,initial,delta,model,body,horizon,hold=hold)
                    ranked.append((score,key,delta,failed,trace,1.))
                best=max(ranked,key=lambda row:row[0])
                if best[3]:
                    first=sorted(ranked,key=lambda row:row[0],reverse=True)[:4]
                    second=[option for option in self.expanded
                            if option[0].startswith(("physical_hips_", "physical_knees_",
                                                      "physical_ankles_", "roll_"))]
                    for _,key,delta,_,_,_ in first:
                        for follow_name,follow in second:
                            score,failed,trace=self.simulate(old,initial,delta,model,body,horizon,
                                                            hold=6,follow=follow)
                            ranked.append((score,f"{key}/{follow_name}",delta,failed,trace,1.))
                    best=max(ranked,key=lambda row:row[0])
            self.applied=best[2]
            self.phase_rate=best[5]
            nominal=ranked[0]
            self.records.append({"time":float(old.time),"x":x,"option":best[1],"phase_rate":best[5],"nominal_score":nominal[0],
                                 "selected_score":best[0],"nominal_would_fall":nominal[3],"selected_would_fall":best[3],
                                 "nominal_final_x":nominal[4][-1][0],"predicted_final_x":best[4][-1][0]})
            self.remaining=6
        self.remaining-=1
        action=np.clip(raw+self.applied,-1.,1.).astype(np.float32)
        self.samples.append((x,np.asarray(obs,np.float32).copy(),raw.copy(),action.copy()))
        return action


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(job):
    (seed,mode,out,start,end,downhill_aware,horizon,exit_vx,stop_x,crest_horizon,crest_speed,
     crest_phase,preview_hold,dynamic_balance,kinematic_choices,recovery_bank,capture,
     crest_pause,descent_vx,snapshot_x)=job
    contract=load_contract(str(CONTRACT))
    policy=RebuiltHeadingPolicy.load(str(BASE),str(CONTRACT))
    policy.warmup()
    cfg=MissionConfig(command_mode="p070",terrain="slope10",goal_xy=(5.,0.),seed=seed,dance=False,
                      max_seconds=160.,stop_x=stop_x,contract=contract,trace=True,render_every=1)
    cfg.profile["uphill"]=.70
    cfg.profile["exit"]=exit_vx
    if descent_vx is not None and mode=="predictive":
        cfg.profile["downhill_transition_vx"]=descent_vx
    planner=None
    if mode=="predictive":
        planner=EntryPlanner(policy,cfg,start,end,downhill_aware,horizon,crest_horizon,
                             crest_speed,crest_phase,preview_hold,dynamic_balance,kinematic_choices,recovery_bank)
        policy.infer=planner.infer
        if crest_phase:
            planner.base_commit=policy.commit
            policy.commit=planner.commit
    spec=GoalSpec("explicit_coordinate",arrival_radius_m=.10)
    poses,tilts=[],[]
    snapshots=[]
    next_snapshot=iter(sorted(snapshot_x))
    boundary=next(next_snapshot,None)
    pause_started=None
    pause_x=None

    def observe(event):
        nonlocal boundary
        if capture:
            poses.append(event["data"].qpos.copy())
            tilts.append(event["tilt"])
        if boundary is not None and event["x"]>=boundary:
            state=np.empty(mujoco.mj_stateSize(event["model"],STATE_SPEC))
            mujoco.mj_getState(event["model"],event["data"],state,STATE_SPEC)
            snapshots.append((boundary,event["x"],state,policy.actions.copy(),policy.targets.copy(),
                              policy.phase,policy.heading,policy.reference_heading,
                              np.asarray(policy.mode,dtype=np.int16)))
            boundary=next(next_snapshot,None)

    def pause_route(x,y,config):
        nonlocal pause_started,pause_x
        if pause_started is None and 2.46<=x<2.66:
            if all(A.contacts()) or x>=2.59:
                pause_started=float(A.data.time)
                pause_x=float(x)
        if pause_started is not None and float(A.data.time)-pause_started<crest_pause:
            planner.remaining=0
            return MotionCommand(0.,0.,0.,1.,"brake")
        return p070_command(x,y,config)

    if mode=="predictive":
        mission.p070_command=pause_route if crest_pause else p070_command
    try:
        row=run_mission(policy,CoordinateGoalAdapter(spec,(5.,0.)),spec,cfg,
                        observer=observe if (capture or snapshot_x) else None)
    finally:
        mission.p070_command=original_p070
    records=[] if planner is None else planner.records
    raw=out/f"{mode}_{seed}_raw.json.gz"
    with gzip.open(raw,"wt",encoding="utf-8") as stream:
        json.dump({"mission":row,"planner":records},stream)
    training_data = None
    if planner is not None and planner.samples:
        xs,obs,base_actions,applied=zip(*planner.samples)
        path=out/f"teacher_{seed}.npz"
        np.savez_compressed(path,x=np.asarray(xs,np.float32),obs=np.asarray(obs,np.float32),
                            base_actions=np.asarray(base_actions,np.float32),
                            actions=np.asarray(applied,np.float32))
        training_data={"file":path.name,"sha256":sha(path),"rows":len(xs)}
    capture_info=None
    if capture:
        p=out/f"{mode}_{seed}_poses.npz"
        np.savez_compressed(p,qpos=np.asarray(poses),tilt=np.asarray(tilts),seed=seed,dt=.02)
        capture_info={"file":p.name,"sha256":sha(p),"frames":len(poses)}
    snapshot_info=None
    if snapshots:
        names=("threshold_x","x","state","actions","targets","phase","heading","reference_heading","mode")
        p=out/f"{mode}_{seed}_continuation.npz"
        np.savez_compressed(p,**{key:np.stack(values) for key,values in zip(names,zip(*snapshots))})
        snapshot_info={"file":p.name,"sha256":sha(p),"points":[float(s[1]) for s in snapshots]}
    return dict(mode=mode,seed=seed,seconds=row["seconds"],success=row["strict_mission_success"],
                fell=row["fell"],collision=row["wall_hit"] or row["body_ground_hit"],
                x_max_m=max(point[0] for point in row["path"]),final_distance_m=row["final_distance_m"],
                decisions=len(records),nonzero_choices=sum(r["option"]!="none" for r in records),
                trace=raw.name,trace_sha256=sha(raw),training_data=training_data,capture=capture_info,
                pause_x_m=pause_x,pause_started_s=pause_started,snapshot=snapshot_info)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--start",type=float,default=.48)
    parser.add_argument("--end",type=float,default=1.35)
    parser.add_argument("--downhill-aware",action="store_true")
    parser.add_argument("--horizon",type=int,default=18)
    parser.add_argument("--crest-horizon",type=int)
    parser.add_argument("--crest-speed",type=float)
    parser.add_argument("--crest-phase",action="store_true")
    parser.add_argument("--preview-hold",type=int,default=6)
    parser.add_argument("--dynamic-balance",action="store_true")
    parser.add_argument("--kinematic-choices",action="store_true")
    parser.add_argument("--recovery-bank",action="store_true")
    parser.add_argument("--capture",action="store_true")
    parser.add_argument("--crest-pause",type=float,default=0.)
    parser.add_argument("--descent-vx",type=float)
    parser.add_argument("--snapshot-x",nargs="+",type=float,default=())
    parser.add_argument("--exit-vx",type=float,default=.07)
    parser.add_argument("--stop-x",type=float,default=5.02)
    parser.add_argument("--seeds",type=int,nargs="+",default=(105140,105141))
    args=parser.parse_args()
    assert mujoco.__version__=="3.6.0"
    args.out.mkdir()
    jobs=[(seed,mode,args.out,args.start,args.end,args.downhill_aware,args.horizon,args.exit_vx,args.stop_x,
           args.crest_horizon,args.crest_speed,args.crest_phase,args.preview_hold,
           args.dynamic_balance,args.kinematic_choices,args.recovery_bank,args.capture,args.crest_pause,
           args.descent_vx,args.snapshot_x)
          for seed in args.seeds for mode in ("accepted","predictive")]
    rows=[]
    with ProcessPoolExecutor(max_workers=min(4,len(jobs))) as pool:
        for row in pool.map(run,jobs):
            print(json.dumps(row),flush=True)
            rows.append(row)
    report={"scope":"Physics lookahead teacher on two development seeds, not a deployment controller",
            "terrain":"slope10","base_sha256":sha(BASE),"controller_window_x":[args.start,args.end],
            "decision_step":6,"horizon_ticks":args.horizon,"downhill_aware":args.downhill_aware,
            "crest_horizon":args.crest_horizon,"crest_speed":args.crest_speed,"crest_phase":args.crest_phase,
            "preview_hold":args.preview_hold,"dynamic_balance":args.dynamic_balance,
            "kinematic_choices":args.kinematic_choices,"recovery_bank":args.recovery_bank,
            "crest_pause_s":args.crest_pause,"descent_vx":args.descent_vx,
            "exit_vx":args.exit_vx,"stop_x":args.stop_x,"action_envelope":.35 if args.recovery_bank else .20,"rows":rows}
    (args.out/"paired.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
