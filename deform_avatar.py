# app/deform_avatar.py — FINAL (multi-photo + FaceMask + 4K + pose-after-bake)
import bpy, os, sys, argparse, math, json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--preset", type=str, default="neutral")
parser.add_argument("--height", type=float, required=True)
parser.add_argument("--chest", type=float)
parser.add_argument("--waist", type=float)
parser.add_argument("--hips", type=float)
parser.add_argument("--shoulder", type=float)
parser.add_argument("--inseam", type=float)
parser.add_argument("--arm", type=float)
parser.add_argument("--frontTex", type=str)
parser.add_argument("--sideTex", type=str)
parser.add_argument("--backTex", type=str)
parser.add_argument("--frontTexList", type=str, default="")
parser.add_argument("--sideTexList",  type=str, default="")
parser.add_argument("--backTexList",  type=str, default="")
parser.add_argument("--texRes", type=int, default=2048)
parser.add_argument("--highDetail", action="store_true")
parser.add_argument("--poseJson", type=str, default="")   # JSON with angles; applied AFTER bake
parser.add_argument("--out", type=str, default="/tmp/avatar.glb")
parser.add_argument("--make_bases", action="store_true")
args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

# Optional: generate placeholder bases
if args.make_bases:
    base_dir = "/app/assets"
    os.makedirs(base_dir, exist_ok=True)
    for cat in ["male","female","neutral","child","baby"]:
        bpy.ops.wm.read_homefile(use_empty=True)
        bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
        bpy.context.active_object.name = f"Base_{cat}"
        bpy.ops.object.shade_smooth()
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(base_dir, f"base_{cat}.blend"))
    print("Base .blend files created.")
    sys.exit(0)

# ---- helpers ----
def _main_mesh():
    ms = [o for o in bpy.data.objects if o.type == "MESH"]
    if not ms:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
        return bpy.context.active_object
    return ms[0]

def _ensure_uv(obj):
    bpy.ops.object.mode_set(mode='OBJECT')
    if not obj.data.uv_layers:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.smart_project(angle_limit=66)
        bpy.ops.object.mode_set(mode='OBJECT')

def _load_img_node(nodes, links, path):
    if not path or not os.path.exists(path): return None
    img = bpy.data.images.load(path)
    tex = nodes.new("ShaderNodeTexImage"); tex.image = img; tex.extension = 'CLIP'
    tcoord = nodes.get("TexCoord") or nodes.new("ShaderNodeTexCoord"); tcoord.name = "TexCoord"
    links.new(tcoord.outputs["Camera"], tex.inputs["Vector"])
    return tex

def _split_list(arg: str):
    return [p for p in (arg or "").split(";") if p and os.path.exists(p)]

def _load_many(nodes, links, paths):
    out = []
    for p in paths:
        n = _load_img_node(nodes, links, p)
        if n: out.append(n)
    return out

def _merge_images(nodes, links, tex_nodes, weights=None):
    if not tex_nodes: return None
    if len(tex_nodes) == 1: return tex_nodes[0].outputs["Color"]
    if not weights or len(weights) != len(tex_nodes):
        weights = [1.0] * len(tex_nodes)
    s = sum(weights) or 1.0
    weights = [w/s for w in weights]
    acc_color = tex_nodes[0].outputs["Color"]; acc_w = weights[0]
    for tex, w in zip(tex_nodes[1:], weights[1:]):
        mix = nodes.new("ShaderNodeMixRGB")
        denom = nodes.new("ShaderNodeMath"); denom.operation='ADD'
        denom.inputs[0].default_value = acc_w; denom.inputs[1].default_value = w
        div = nodes.new("ShaderNodeMath"); div.operation='DIVIDE'
        links.new(denom.outputs["Value"], div.inputs[1]); div.inputs[0].default_value = w
        links.new(div.outputs["Value"], mix.inputs["Fac"])
        links.new(acc_color, mix.inputs[1]); links.new(tex.outputs["Color"], mix.inputs[2])
        acc_color = mix.outputs["Color"]; acc_w += w
    return acc_color

def _make_facemask_if_missing(obj, name="FaceMask"):
    if obj.data.attributes.get(name): return name
    attr = obj.data.attributes.new(name=name, type='FLOAT', domain='POINT')
    verts = obj.data.vertices
    if len(verts) == 0: return name
    zs = np.array([v.co.z for v in verts], dtype=np.float32)
    z_min, z_max = float(zs.min()), float(zs.max())
    z_low = z_min + 0.65 * (z_max - z_min); z_high = z_min + 0.95 * (z_max - z_min)
    obj.data.calc_normals()
    vals = np.zeros(len(verts), dtype=np.float32)
    for i, v in enumerate(verts):
        ny = v.normal.y; z = v.co.z
        head_band = 0.0
        if z >= z_low: head_band = min(1.0, max(0.0, (z - z_low) / max(1e-6, (z_high - z_low))))
        frontness = 1.0 if ny < -0.15 else 0.0
        vals[i] = float(head_band * frontness)
    if vals.max() > 0: vals = (vals / vals.max()) ** 0.7
    attr_data = obj.data.attributes[name].data
    for i, v in enumerate(attr_data): v.value = float(vals[i])
    return name

def _set_simple_shape(obj):
    scale = max(0.2, float(args.height) / 1.75)
    obj.scale = (scale, scale, scale)

def ensure_basic_armature(body_obj):
    """Create a minimal armature and bind if none exists."""
    arm = None
    for o in bpy.data.objects:
        if o.type == 'ARMATURE':
            arm = o; break
    if not arm:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.armature_add(enter_editmode=True)
        arm = bpy.context.active_object
        arm.name = "Armature"
        eb = arm.data.edit_bones
        spine = eb[0]; spine.name = "spine"
        spine.head = (0, 0, 0.9); spine.tail = (0, 0, 1.4)
        head = eb.new("head"); head.head = spine.tail; head.tail = (0, 0, 1.7)
        l_up = eb.new("upper_arm.L"); l_up.head=(0.05,0,1.35); l_up.tail=(0.35,0,1.35)
        r_up = eb.new("upper_arm.R"); r_up.head=(-0.05,0,1.35); r_up.tail=(-0.35,0,1.35)
        l_fk = eb.new("forearm.L"); l_fk.head=l_up.tail; l_fk.tail=(0.55,0,1.30)
        r_fk = eb.new("forearm.R"); r_fk.head=r_up.tail; r_fk.tail=(-0.55,0,1.30)
        bpy.ops.object.mode_set(mode='OBJECT')
        # Parent body to armature with automatic weights
        bpy.context.view_layer.objects.active = body_obj
        body_obj.select_set(True); arm.select_set(True)
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        body_obj.select_set(False); arm.select_set(False)
    return arm

def apply_pose_from_angles(arm, angles: dict):
    """Apply rough pose from angles (deg)."""
    if not angles: return
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    pb = arm.pose.bones
    def rot_deg(name, axis, deg):
        if name not in pb or deg is None: return
        b = pb[name]; b.rotation_mode = 'XYZ'
        e = list(b.rotation_euler)
        ax = {"X":0,"Y":1,"Z":2}[axis]
        d = max(-120.0, min(120.0, float(deg)))
        e[ax] = math.radians(d); b.rotation_euler = e
    rot_deg("upper_arm.L", "Z",  angles.get("left_shoulder_abd"))
    rot_deg("upper_arm.R", "Z", -angles.get("right_shoulder_abd"))
    rot_deg("forearm.L",  "Y",  -(180.0 - angles.get("left_elbow", 180.0)))
    rot_deg("forearm.R",  "Y",   (180.0 - angles.get("right_elbow", 180.0)))
    rot_deg("head",       "Z",   angles.get("head_yaw"))
    bpy.ops.object.mode_set(mode='OBJECT')

# ---- scene & model ----
obj = _main_mesh(); obj.name = "Avatar"
bpy.context.view_layer.objects.active = obj
_set_simple_shape(obj); _ensure_uv(obj)

# ---- material & nodes ----
mat_name = "AvatarProjection"
mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
mat.use_nodes = True
nodes = mat.node_tree.nodes; links = mat.node_tree.links
nodes.clear()

out = nodes.new("ShaderNodeOutputMaterial")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

frontN = _load_img_node(nodes, links, args.frontTex)
sideN  = _load_img_node(nodes, links, args.sideTex)
backN  =
