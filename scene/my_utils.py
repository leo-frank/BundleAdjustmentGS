import torch
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
class Pose():
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self,R=None,t=None):
        # construct a camera pose from the given R and/or t
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3,device=t.device).repeat(*t.shape[:-1],1,1)
        elif t is None:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1],device=R.device)
        else:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3,3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R,t[...,None]],dim=-1) # [...,3,4]
        assert(pose.shape[-2:]==(3,4))
        return pose

    def invert(self,pose,use_inverse=False):
        # invert a camera pose
        R,t = pose[...,:3],pose[...,3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[...,0]
        pose_inv = self(R=R_inv,t=t_inv)
        return pose_inv

    def compose(self,pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new,pose)
        return pose_new

    def compose_pair(self,pose_a,pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a,t_a = pose_a[...,:3],pose_a[...,3:]
        R_b,t_b = pose_b[...,:3],pose_b[...,3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[...,0]
        pose_new = self(R=R_new,t=t_new)
        return pose_new

def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

def cam2world(X,pose):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom@pose_inv.transpose(-1,-2)

def get_camera_mesh(pose,depth=1):
    vertices = torch.tensor([[-0.5,-0.5,1],
                             [0.5,-0.5,1],
                             [0.5,0.5,1],
                             [-0.5,0.5,1],
                             [0,0,0]])*depth
    faces = torch.tensor([[0,1,2],
                          [0,2,3],
                          [0,1,4],
                          [1,2,4],
                          [2,3,4],
                          [3,0,4]])
    vertices = cam2world(vertices[None],pose)
    wireframe = vertices[:,[0,1,2,3,0,4,1,2,4,3]]
    return vertices,faces,wireframe


def setup_3D_plot(ax,elev,azim,lim=None):
    ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.yaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.zaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_xlabel("X",fontsize=16)
    ax.set_ylabel("Y",fontsize=16)
    ax.set_zlabel("Z",fontsize=16)
    ax.set_xlim(lim.x[0],lim.x[1])
    ax.set_ylim(lim.y[0],lim.y[1])
    ax.set_zlim(lim.z[0],lim.z[1])
    ax.view_init(elev=elev,azim=azim)

def plot_save_poses_blender(pose, pose_ref=None,file_path=None,ep=None):
    '''
    both are w2c, in colmap system
    '''
    fig = plt.figure(figsize=(10,10))
    if isinstance(pose, np.ndarray): pose = torch.tensor(pose)
    if pose_ref is not None and isinstance(pose_ref, np.ndarray): pose_ref = torch.tensor(pose_ref)
    pose = pose[..., :3, :4]
    pose_ref = pose_ref[..., :3, :4]
    # get the camera meshes
    _,_,cam = get_camera_mesh(pose[..., :3, :4],depth=0.5)
    cam = cam.numpy()
    if pose_ref is not None:
        _,_,cam_ref = get_camera_mesh(pose_ref,depth=0.5)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    ax = fig.add_subplot(111,projection="3d")
    ax.set_title("epoch {}".format(ep),pad=0)
    setup_3D_plot(ax,elev=45,azim=35,lim=edict(x=(-3,3),y=(-3,3),z=(-3,2.4)))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    # plot the cameras
    N = len(cam)
    ref_color = (0.7,0.2,0.7)
    pred_color = (0,0.6,0.7)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam_ref],alpha=0.2,facecolor=ref_color))
    for i in range(N):
        ax.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=ref_color,linewidth=0.5)
        ax.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=ref_color,s=20)
    if ep==0:
        png_fname = "{}/GT.png".format(file_path)
        plt.savefig(png_fname,dpi=75)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam],alpha=0.2,facecolor=pred_color))
    for i in range(N):
        ax.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=pred_color,linewidth=1)
        ax.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=pred_color,s=20)
    for i in range(N):
        ax.plot([cam[i,5,0],cam_ref[i,5,0]],
                [cam[i,5,1],cam_ref[i,5,1]],
                [cam[i,5,2],cam_ref[i,5,2]],color=(1,0,0),linewidth=3)
    png_fname = "{}".format(file_path,ep)
    plt.savefig(png_fname,dpi=75)
    # clean up
    plt.clf()


def procrustes_analysis(X0,X1): # [N,3]
    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U,S,V = (X0cs.t()@X1cs).double().svd(some=True)
    R = (U@V.t())
    if R.det()<0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0],t1=t1[0],s0=s0,s1=s1,R=R)
    return sim3

def prealign_cameras(pose,pose_GT):
    if isinstance(pose, np.ndarray): pose = torch.tensor(pose)
    if isinstance(pose_GT, np.ndarray): pose_GT = torch.tensor(pose_GT)
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1,1,3)
    center_pred = cam2world(center,pose)[:,0] # [N,3]
    center_GT = cam2world(center,pose_GT)[:,0] # [N,3]
    center_pred = center_pred.double()
    center_GT = center_GT.double()
    try:
        sim3 = procrustes_analysis(center_GT,center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3))
        raise Exception("SVD did not converge")
    # align the camera poses
    center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
    R_aligned = pose[...,:3]@sim3.R.t()
    t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
    pose_aligned = torch.cat([R_aligned,t_aligned[...,None]],dim=-1)
    return pose_aligned,sim3

def rotation_distance(R1,R2,eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

def evaluate_camera_alignment(pose_aligned,pose_GT):
    if isinstance(pose_aligned, np.ndarray): pose_aligned = torch.tensor(pose_aligned)
    if isinstance(pose_GT, np.ndarray): pose_GT = torch.tensor(pose_GT)
    # measure errors in rotation and translation
    R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
    R_GT,t_GT = pose_GT.split([3,1],dim=-1)
    R_error = rotation_distance(R_aligned,R_GT)
    t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
    error = edict(R=R_error,t=t_error)
    return error