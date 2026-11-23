"""Fit a downhill feedback controller from real native simulation states."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from slope10_predictive_snapshots_20261123 import (
    A, BASE, CONTRACT, MissionConfig, RebuiltHeadingPolicy, STATE_SPEC,
    load_contract, mujoco, np, p070_command, sha,
)
import r4_profile as R


NAMES = [
    "hip_symmetric", "hip_stance", "knee_symmetric", "knee_differential", "knee_swing",
    "ankle_symmetric", "ankle_differential", "ankle_stance", "ankle_speed",
    "pitch_hip", "pitch_rate_hip", "roll", "roll_rate", "cadence", "speed",
]
LIMIT = .40
HIP_AXIS = np.array([1.,-1.])
SIDE = np.array([-1.,1.])
DT = .02


class DownhillFeedback:
    def __init__(self, policy, gains):
        self.policy = policy
        self.gains = gains
        self.original = policy.infer
        self.apply = policy.commit
        self.phase_rate = 1.

    def command(self, command):
        x=float(A.data.qpos[0])
        weight=min(1.,max(0.,(x-2.35)/.15),max(0.,(4.27-x)/.15))
        forward=(float(np.clip(command.vx+.026*self.gains[-1]*weight,.011,.078))
                 if weight else command.vx)
        return np.array((forward,command.vy,command.yaw_rate),dtype=np.float32)

    def infer(self, obs):
        raw=self.original(obs)
        x=float(A.data.qpos[0])
        weight=min(1.,max(0.,(x-2.30)/.14),max(0.,(4.35-x)/.16))
        if weight==0:
            self.phase_rate=1.
            return raw
        g=self.gains
        r=A.data.site_xmat[A.imu].reshape(3,3)
        pitch=float(np.arcsin(np.clip(-r[2,0],-1.,1.)))
        roll=float(np.arctan2(r[2,1],r[2,2]))
        pitch_error=np.clip((pitch-.44)/.32,-1.,1.)
        gyro=np.clip(obs[:3]/4.,-1.,1.)
        support=(obs[69:71]>.5).astype(np.float32)
        excess=np.clip((float(A.data.qvel[0])-.10)/.22,-1.,1.)
        delta=np.zeros(10,np.float32)
        delta[[2,7]]=HIP_AXIS*(.23*g[0]+.22*g[1]*support+.32*g[9]*pitch_error+.28*g[10]*gyro[1])
        delta[[3,8]]=.23*g[2]+.17*g[3]*SIDE+.25*g[4]*(1.-support)
        delta[[4,9]]=.25*g[5]+.20*g[6]*SIDE+.22*g[7]*support+.20*g[8]*excess*support
        delta[[1,6]]=.25*(g[11]*np.clip(roll/.28,-1.,1.)+g[12]*gyro[0])
        self.phase_rate=1.+.33*float(g[13])*weight
        return np.clip(raw+weight*np.clip(delta,-LIMIT,LIMIT),-1.,1.)

    def commit(self, action, target):
        phase=self.policy.phase
        self.apply(action,target)
        self.policy.phase=(phase+self.phase_rate)%self.policy.contract["phase_period_steps"]


CONTEXT = {}


def init():
    R.init_custom("slope10")
    CONTEXT["model"]=A.model
    CONTEXT["contract"]=load_contract(str(CONTRACT))


def rollout(job):
    i, source, snapshot, gains, duration=job
    model=CONTEXT["model"]
    A.data=mujoco.MjData(model)
    mujoco.mj_setState(model,A.data,snapshot["state"],STATE_SPEC)
    mujoco.mj_forward(model,A.data)
    policy=RebuiltHeadingPolicy.load(str(BASE),str(CONTRACT))
    for k in ("actions","targets"):
        setattr(policy,k,snapshot[k].copy())
    for k in ("phase","heading","reference_heading"):
        setattr(policy,k,float(snapshot[k]))
    policy.mode=tuple(int(n) for n in snapshot["mode"])
    restored_targets_error=float(np.max(np.abs(A.data.ctrl-policy.targets)))
    if restored_targets_error>1e-6:
        raise ValueError(f"native ctrl and actor history mismatch {restored_targets_error}")
    feedback=DownhillFeedback(policy,gains)
    config=MissionConfig(command_mode="p070",terrain="slope10",goal_xy=(5.,0.),
                         contract=CONTEXT["contract"],trace=False,dance=False,stop_x=4.96)
    config.profile["uphill"],config.profile["exit"]=.70,.10
    initial=float(A.data.qpos[0])
    source_distance_error=abs(initial-float(snapshot["x"]))
    if source_distance_error>1e-9:
        raise ValueError("Invalid restored position")
    max_x=initial
    peak_tilt=0.
    unstable=0.
    ended=False
    for step in range(int(round(duration/DT))):
        command=p070_command(A.data.qpos[0],A.data.qpos[1],config)
        obs=policy.build(A.sensor("gyro"),A.sensor("accelerometer"),feedback.command(command),
                         A.data.qpos[A.qaddr],A.data.qvel[A.daddr],A.contacts(),heading=A.heading())
        action=feedback.infer(obs)
        target=policy.target(action)
        feedback.commit(action,target)
        if np.max(np.abs(A.data.ctrl-target))>5.24*.02+1e-6:
            raise ValueError("Joint slew envelope exceeded")
        A.data.ctrl[:]=target
        for _ in range(10):
            mujoco.mj_step(model,A.data)
        mujoco.mj_forward(model,A.data)
        x=float(A.data.qpos[0])
        max_x=max(x,max_x)
        up=A.data.site_xmat[A.imu].reshape(3,3)[2,2]
        tilt=float(np.degrees(np.arccos(np.clip(up,-1.,1.))))
        peak_tilt=max(peak_tilt,tilt)
        unstable+=max(tilt-42.,0.)/36.
        ended=bool(A.data.qpos[2]<.05 or tilt>75. or A.wall_hit() or A.body_ground_hit())
        if ended:
            break
    final=float(A.data.qpos[0])
    time=(step+1)*DT
    score=1.6*(max_x-initial)+1.1*(final-initial)+.055*time-.65*ended
    score-=.25*abs(float(A.data.qpos[1]))+.07*unstable/(step+1)
    return dict(i=i,source=source,start_x=initial,max_x=max_x,end_x=final,
                tilt=peak_tilt,seconds=time,failed=ended,score=score)


def sources(files):
    starts=[]
    for file in files:
        with np.load(file,allow_pickle=False) as states:
            for i,x in enumerate(states["x"]):
                starts.append((file.name,{key:states[key][i] for key in states.files}))
    if len(starts)<5:
        raise ValueError("Expected physically recorded crest snapshots")
    return starts


def run_batch(pool, starts, variants, duration):
    return list(pool.map(
        rollout,
        [(i,label,state,gains,duration) for i,gains in enumerate(variants)
         for label,state in starts],
        chunksize=2,
    ))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--snapshots",type=Path,nargs="+",required=True)
    parser.add_argument("--rounds",type=int,default=7)
    parser.add_argument("--population",type=int,default=24)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    assert mujoco.__version__=="3.6.0"
    args.out.mkdir()
    full=sources(args.snapshots)
    # Keep alternate real observations separate from population selection.
    train=full[::2]
    valid=full[1::2]
    rng=np.random.default_rng(431001)
    mean=np.zeros(len(NAMES),np.float32)
    std=np.full_like(mean,.43)
    best=mean.copy()
    best_score=-np.inf
    history=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init) as pool:
        for round_index in range(args.rounds):
            variants=rng.normal(mean,std,size=(args.population,len(NAMES))).astype(np.float32)
            np.clip(variants,-1.,1.,out=variants)
            variants[0],variants[1]=best,0.
            results=run_batch(pool,train,variants,7.)
            scores=np.asarray([r["score"] for r in results]).reshape((len(variants),len(train)))
            robust=.65*scores.mean(axis=1)+.35*scores.min(axis=1)
            selected=np.argsort(robust)
            elite=variants[selected[-5:]]
            mean=.35*mean+.65*elite.mean(axis=0)
            std=np.clip(.35*std+.65*elite.std(axis=0),.11,.60)
            i=int(selected[-1])
            if robust[i]>=best_score:
                best_score=float(robust[i])
                best=variants[i].copy()
            row=dict(round=round_index+1,episodes=len(results),score=best_score,
                     samples=[r for r in results if r["i"]==i])
            history.append(row)
            (args.out/"train.json").write_text(json.dumps(history,indent=2)+"\n")
            print(json.dumps(row),flush=True)
        check=run_batch(pool,valid,[np.zeros_like(best),best],10.)
    actor=args.out/"downhill_feedback.npz"
    np.savez_compressed(actor,gains=best,names=NAMES)
    report=dict(
        terrain="slope10", scope="Native continuation from alternate saved states",
        envelope=LIMIT, train_episodes=sum(r["episodes"] for r in history),
        actor_sha256=sha(BASE), weights_sha256=sha(actor),
        snapshots=[dict(path=str(p),sha256=sha(p)) for p in args.snapshots],
        validation=check,
    )
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
