"""Replay an Isaac Lab Go2 policy in MuJoCo (sim2sim).

Usage examples:
    python prueba_v2.py --order type
    python prueba_v2.py --order leg
    python prueba_v2.py --order leg_fr_first
    python prueba_v2.py --joints FL_hip_joint,FL_thigh_joint,...   (exact list printed by Isaac Lab)
"""
import os
os.environ["MUJOCO_GL"] = "egl"          # offscreen rendering, no display needed
import argparse
import numpy as np
import mujoco
import imageio

# ----------------------------------------------------------------------------
# Joint orders to try (the one Isaac Lab used in training MUST be matched)
# ----------------------------------------------------------------------------
LEGS = ["FL", "FR", "RL", "RR"]
ORDERS = {
    # all hips, then all thighs, then all calves (PhysX-style)
    "type": [f"{l}_{j}_joint" for j in ("hip", "thigh", "calf") for l in LEGS],
    # leg by leg: FL, FR, RL, RR
    "leg": [f"{l}_{j}_joint" for l in LEGS for j in ("hip", "thigh", "calf")],
    # leg by leg, Unitree SDK leg order: FR, FL, RR, RL
    "leg_fr_first": [f"{l}_{j}_joint" for l in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf")],
}

# ----------------------------------------------------------------------------
# Values taken from params/env.yaml
# ----------------------------------------------------------------------------
KP, KD = 25.0, 0.5                       # actuators.base_legs stiffness / damping
ACTION_SCALE = 0.25                      # actions.joint_pos.scale (use_default_offset: true)
SIM_DT = 0.005                           # sim.dt
DECIMATION = 4                           # decimation
INIT_HEIGHT = 0.40                       # init_state.pos z
SCAN_N = 187             # 17 x 11 grid, check env.yaml
SCAN_OFFSET = 0.5        # height_scan offset, check env.yaml
SCAN_CLIP = 1.0

def limits(name):
    """(effort_limit = saturation_effort, velocity_limit) from the DCMotor config."""
    if "calf" in name:
        return 45.43, 15.7
    return 23.7, 30.1                    # hip and thigh


def default_pos(name):
    """init_state.joint_pos, looked up by joint NAME."""
    front, left = name.startswith("F"), name[1] == "L"
    if "hip" in name:
        return 0.1 if left else -0.1
    if "thigh" in name:
        return 0.8 if front else 1.0
    return -1.5                          # calf


ap = argparse.ArgumentParser()
#ap.add_argument("--policy", default="./policy_300_it.pt")          # .pt (TorchScript) or .onnx
ap.add_argument("--policy", default="./policy.pt")          # .pt (TorchScript) or .onnx
ap.add_argument("--model", default="~/mujoco_menagerie/unitree_go2/scene.xml")
ap.add_argument("--order", default="type", choices=list(ORDERS))
ap.add_argument("--joints", default=None, help="comma-separated exact joint order (overrides --order)")
ap.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0], help="vx vy yaw_rate")
ap.add_argument("--steps", type=int, default=500)           # policy steps (50 Hz -> 10 s)
ap.add_argument("--substeps", type=int, default=2, help="MuJoCo steps per sim.dt (env.yaml num_substeps: 2)")
ap.add_argument("--out", default=None)
args = ap.parse_args()

JOINTS = args.joints.split(",") if args.joints else ORDERS[args.order]
assert len(JOINTS) == 12, "need 12 joint names"
OUT = args.out or f"go2_play_{args.order if not args.joints else 'custom'}.mp4"
CMD = np.array(args.cmd)

DEFAULT = np.array([default_pos(n) for n in JOINTS])
EFFORT = np.array([limits(n)[0] for n in JOINTS])
VMAX = np.array([limits(n)[1] for n in JOINTS])

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
args.model = os.path.expanduser(args.model)
args.policy = os.path.expanduser(args.policy)

model = mujoco.MjModel.from_xml_path(args.model)
data = mujoco.MjData(model)

# Solver settings aligned with env.yaml (mujoco_warp: newton solver, implicitfast, pyramidal cone)
model.opt.timestep = SIM_DT / args.substeps
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
model.opt.iterations = 100
model.opt.ls_iterations = 50
model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
model.opt.impratio = 1.0
model.opt.tolerance = 1e-6
model.geom_friction[:, 0] = 1.0          # physics_material friction 1.0

qadr, vadr = [], []
for name in JOINTS:
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert j >= 0, f"joint {name} not found in the MuJoCo model"
    qadr.append(model.jnt_qposadr[j])
    vadr.append(model.jnt_dofadr[j])
qadr, vadr = np.array(qadr), np.array(vadr)

# Ignore Menagerie's own actuators; apply an explicit DC-motor PD torque instead
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
model.dof_damping[vadr] = 0.0
model.dof_frictionloss[vadr] = 0.0


def pd_torque(target):
    """Isaac Lab DCMotor: PD torque clipped by a linear torque-speed curve."""
    q, qd = data.qpos[qadr], data.qvel[vadr]
    tau = KP * (target - q) - KD * qd
    top = np.clip(EFFORT * (1.0 - qd / VMAX), 0.0, EFFORT)
    bot = np.clip(EFFORT * (-1.0 - qd / VMAX), -EFFORT, 0.0)
    return np.clip(tau, bot, top)


# ----------------------------------------------------------------------------
# Policy (TorchScript or ONNX)
# ----------------------------------------------------------------------------
if args.policy.endswith(".onnx"):
    import onnxruntime as ort
    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def act_fn(o):
        return sess.run(None, {in_name: o[None]})[0][0]
else:
    import torch
    policy = torch.jit.load(args.policy).eval()

    def act_fn(o):
        with torch.no_grad():
            return policy(torch.from_numpy(o)[None])[0].numpy()


# ----------------------------------------------------------------------------
# Reset and observations (layout from env.yaml: lin_vel, ang_vel, gravity,
# command, joint_pos_rel, joint_vel_rel, last_action = 48 values, no scales)
# ----------------------------------------------------------------------------
mujoco.mj_resetData(model, data)
data.qpos[0:3] = [0.0, 0.0, INIT_HEIGHT]
data.qpos[3:7] = [1, 0, 0, 0]            # MuJoCo quaternion is w,x,y,z
data.qpos[qadr] = DEFAULT
mujoco.mj_forward(model, data)


def get_obs(last_act):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    R = R.reshape(3, 3)
    lin_b = R.T @ data.qvel[0:3]          # free-joint linear velocity: world frame -> body
    ang_b = data.qvel[3:6]                # free-joint angular velocity: already body frame
    grav_b = R.T @ np.array([0.0, 0.0, -1.0])
    q = data.qpos[qadr] - DEFAULT
    qd = data.qvel[vadr]
    base_obs = np.concatenate([lin_b, ang_b, grav_b, CMD, q, qd, last_act])
    scan = np.full(SCAN_N, np.clip(data.qpos[2] - SCAN_OFFSET, -SCAN_CLIP, SCAN_CLIP))
    return np.concatenate([base_obs, scan]).astype(np.float32)


# ----------------------------------------------------------------------------
# Rollout + video
# ----------------------------------------------------------------------------
renderer = mujoco.Renderer(model, 480, 640)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
cam.distance, cam.elevation, cam.azimuth = 2.0, -20, 135

last_act = np.zeros(12)
fell_at = None
vx_hist = []
start_xy = data.qpos[0:2].copy()
writer = imageio.get_writer(OUT, fps=int(round(1.0 / (SIM_DT * DECIMATION))))
for step in range(args.steps):
    o = get_obs(last_act)
    assert o.shape[0] == 235, f"obs size {o.shape[0]} != 235"
    act = act_fn(o)
    if step < 2:
        print(f"step {step}: act min/max {act.min():+.2f} / {act.max():+.2f}")
    last_act = act

    target = DEFAULT + ACTION_SCALE * act
    for _ in range(DECIMATION * args.substeps):
        data.qfrc_applied[vadr] = pd_torque(target)
        mujoco.mj_step(model, data)

    height = data.qpos[2]
    vx_hist.append(o[0])
    if step % 50 == 0:
        print(f"step {step:4d}  base height {height:.3f}  vx_body {o[0]:+.2f}")
    if fell_at is None and height < 0.15:
        fell_at = step
        print(f"-> base height below 0.15 m at step {step}: the robot fell")

    renderer.update_scene(data, camera=cam)
    writer.append_data(renderer.render())
writer.close()

dist = np.linalg.norm(data.qpos[0:2] - start_xy)
print(f"saved {OUT}")
print(f"order={args.order if not args.joints else 'custom'} | fell at step: {fell_at} | "
      f"distance travelled: {dist:.2f} m | mean vx_body (2nd half): {np.mean(vx_hist[len(vx_hist)//2:]):+.2f} "
      f"(command {CMD[0]:+.2f})")
