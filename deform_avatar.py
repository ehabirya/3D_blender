# Runs INSIDE Blender: deform + project + blend + BAKE → GLB
import bpy, os, sys, argparse, math
import numpy as np

# -------------------------------
# Args
# -------------------------------
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
parser.add_argument("--texRes", type=int, default=2048)
parser.add_argument("--highDetail", action="store_true")  # <— 4K premium switch
parser.add_argument("--out", type=str, default="/tmp/avatar.glb")
parser.add_argument("--make_bases", action="store_true")
args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

# -------------------------------
# Optional: generate placeholder bases
# -------------------------------
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

print(f"▶ deform_avatar: preset={args.preset} height={args.height} highDetail={args.highDetail}")

# -------------------------------
# Helpers
# -------------------------------
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
    if not path or not os.path.exists(path):
        return None
    img = bpy.data.images.load(path)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = 'CLIP'
    # drive by camera coords for projector-like wrap
    tcoord = nodes.get("TexCoord") or nodes.new("ShaderNodeTexCoord"); tcoord.name = "TexCoord"
    links.new(tcoord.outputs["Camera"], tex.inputs["Vector"])
    return tex

def _make_facemask_if_missing(obj, name="FaceMask"):
    # If a float POINT attribute named FaceMask exists, keep it.
    if obj.data.attributes.get(name):
        return name
    # Create a procedural face mask by geometry cues (front-facing + higher head region)
    attr = obj.data.attributes.new(name=name, type='FLOAT', domain='POINT')
    # Build quick spatial ranges for normalization
    verts = obj.data.vertices
    if len(verts) == 0:
        return name
    # work in object space; approximate head by top 35% Z and front-facing normals (ny < -0.15)
    zs = np.array([v.co.z for v in verts], dtype=np.float32)
    z_min, z_max = float(zs.min()), float(zs.max())
    z_low = z_min + 0.65 * (z_max - z_min)   # lower threshold ~ neck
    z_high = z_min + 0.95 * (z_max - z_min)  # head top band

    # need vertex normals in object space
    obj.data.calc_normals()
    # write attribute (domain POINT)
    vals = np.zeros(len(verts), dtype=np.float32)
    for i, v in enumerate(verts):
        nz = v.normal.z
        ny = v.normal.y
        z = v.co.z
        # mask components
        head_band = 0.0
        if z >= z_low:
            # smoothstep between z_low..z_high
            t = min(1.0, max(0.0, (z - z_low) / max(1e-6, (z_high - z_low))))
            head_band = t
        frontness = 1.0 if ny < -0.15 else 0.0
        vals[i] = float(head_band * frontness)
    # normalize & soften a bit
    if vals.max() > 0:
        vals = (vals / vals.max()) ** 0.7
    # store to attribute
    attr_data = obj.data.attributes[name].data
    for i, v in enumerate(attr_data):
        v.value = float(vals[i])
    return name

def _set_simple_shape(obj):
    # Height scaling (assume base ≈ 1.75 m)
    scale = max(0.2, float(args.height) / 1.75)
    obj.scale = (scale, scale, scale)
    # (Optional) simple proportional cues from chest/waist/hips if provided
    # You can replace with shape keys later
    # No-op by default to avoid distorting unknown meshes

# -------------------------------
# Scene / mesh
# -------------------------------
obj = _main_mesh()
obj.name = "Avatar"
bpy.context.view_layer.objects.active = obj
_set_simple_shape(obj)
_ensure_uv(obj)

# -------------------------------
# Build material nodes
# -------------------------------
mat_name = "AvatarProjection"
mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

# Core nodes
out = nodes.new("ShaderNodeOutputMaterial")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

# Try to load images
frontN = _load_img_node(nodes, links, args.frontTex)
sideN  = _load_img_node(nodes, links, args.sideTex)
backN  = _load_img_node(nodes, links, args.backTex)

# Geometry / normals for weighting
geom = nodes.new("ShaderNodeNewGeometry")
sep = nodes.new("ShaderNodeSeparateXYZ")
links.new(geom.outputs["Normal"], sep.inputs["Vector"])

# FRONT weight ≈ clamp(-Ny, 0, 1)
mul_negY = nodes.new("ShaderNodeMath"); mul_negY.operation='MULTIPLY'; mul_negY.inputs[1].default_value = -1.0
links.new(sep.outputs["Y"], mul_negY.inputs[0])
front_w = nodes.new("ShaderNodeMath"); front_w.operation='MAXIMUM'; front_w.inputs[1].default_value = 0.0
links.new(mul_negY.outputs["Value"], front_w.inputs[0])

# BACK weight ≈ clamp(+Ny, 0, 1)
back_w = nodes.new("ShaderNodeMath"); back_w.operation='MAXIMUM'; back_w.inputs[1].default_value = 0.0
links.new(sep.outputs["Y"], back_w.inputs[0])

# SIDE weight ≈ |Nx| * 0.8
abs_x = nodes.new("ShaderNodeMath"); abs_x.operation='ABSOLUTE'
links.new(sep.outputs["X"], abs_x.inputs[0])
side_gain = nodes.new("ShaderNodeMath"); side_gain.operation='MULTIPLY'; side_gain.inputs[1].default_value = 0.8
links.new(abs_x.outputs["Value"], side_gain.inputs[0])

# Mix front & side
mix_fs = nodes.new("ShaderNodeMixRGB")
# Fac = front_w / (front_w + side_w)
sum_fs = nodes.new("ShaderNodeMath"); sum_fs.operation='ADD'
links.new(front_w.outputs["Value"], sum_fs.inputs[0])
links.new(side_gain.outputs["Value"], sum_fs.inputs[1])
div_f = nodes.new("ShaderNodeMath"); div_f.operation='DIVIDE'
links.new(front_w.outputs["Value"], div_f.inputs[0])
links.new(sum_fs.outputs["Value"], div_f.inputs[1])
links.new(div_f.outputs["Value"], mix_fs.inputs["Fac"])
if frontN and sideN:
    links.new(frontN.outputs["Color"], mix_fs.inputs[1])
    links.new(sideN.outputs["Color"],  mix_fs.inputs[2])

# Mix (front/side) with back
mix_all = nodes.new("ShaderNodeMixRGB")
sum_all = nodes.new("ShaderNodeMath"); sum_all.operation='ADD'
links.new(back_w.outputs["Value"], sum_all.inputs[0])
links.new(side_gain.outputs["Value"], sum_all.inputs[1])
div_b = nodes.new("ShaderNodeMath"); div_b.operation='DIVIDE'
links.new(back_w.outputs["Value"], div_b.inputs[0])
links.new(sum_all.outputs["Value"],   div_b.inputs[1])
links.new(div_b.outputs["Value"], mix_all.inputs["Fac"])
if frontN and sideN:
    links.new(mix_fs.outputs["Color"], mix_all.inputs[1])
elif frontN:
    links.new(frontN.outputs["Color"], mix_all.inputs[1])
elif sideN:
    links.new(sideN.outputs["Color"], mix_all.inputs[1])

if backN:
    links.new(backN.outputs["Color"], mix_all.inputs[2])

# -------------------------------
# FaceMask blending (seam killer on face)
# -------------------------------
# 1) ensure a FaceMask attribute (POINT Float). If missing, create procedurally.
fm_name = _make_facemask_if_missing(obj, "FaceMask")

# 2) bring it into the shader
attr = nodes.new("ShaderNodeAttribute"); attr.attribute_name = fm_name

# 3) If we have a front texture, bias towards it on the face:
#    BaseColor := mix( (frontColor), (mix_all result), FaceMask )
if frontN:
    mix_face = nodes.new("ShaderNodeMixRGB")
    links.new(attr.outputs["Fac"], mix_face.inputs["Fac"])
    links.new(frontN.outputs["Color"], mix_face.inputs[1])  # Color1 = face/front
    # source for non-face: if we had any blend result use it; else use front anyway
    if (frontN and sideN) or backN:
        links.new(mix_all.outputs["Color"], mix_face.inputs[2])
    else:
        links.new(frontN.outputs["Color"], mix_face.inputs[2])
    links.new(mix_face.outputs["Color"], bsdf.inputs["Base Color"])
else:
    # fallback: use whatever we had
    if (frontN or sideN or backN):
        links.new(mix_all.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        # ultra fallback: neutral color so poor photos still look clean
        bsdf.inputs["Base Color"].default_value = (0.78, 0.72, 0.68, 1.0)

# Slightly nicer skin defaults if images are weak
bsdf.inputs["Roughness"].default_value = 0.65
bsdf.inputs["Specular"].default_value = 0.35

# Assign material
obj.data.materials.clear()
obj.data.materials.append(mat)

# -------------------------------
# BAKE (color only) with premium 4K switch
# -------------------------------
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'CPU'

# Resolve bake resolution:
res = int(args.texRes)
if args.highDetail:
    res = max(res, 4096)              # premium forces ≥ 4K
else:
    res = max(1024, min(res, 4096))   # keep sane bounds

bake_img = bpy.data.images.get("BakedTex") or bpy.data.images.new("BakedTex", width=res, height=res, alpha=True)
bake_node = nodes.new("ShaderNodeTexImage"); bake_node.image = bake_img
nodes.active = bake_node

_ensure_uv(obj)

# Bake DIFFUSE color only
bpy.context.view_layer.objects.active = obj
for o in bpy.context.selected_objects: o.select_set(False)
obj.select_set(True)

bpy.context.scene.cycles.bake_type = 'DIFFUSE'
bpy.context.scene.render.use_bake_multires = False
bpy.context.scene.render.bake.use_pass_direct = False
bpy.context.scene.render.bake.use_pass_indirect = False
bpy.context.scene.render.bake.margin = 4
bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})

# Replace graph with baked texture for lightweight GLB
nodes.clear()
out2 = nodes.new("ShaderNodeOutputMaterial")
bsdf2 = nodes.new("ShaderNodeBsdfPrincipled")
tex_baked = nodes.new("ShaderNodeTexImage"); tex_baked.image = bake_img
links = mat.node_tree.links
links.new(tex_baked.outputs["Color"], bsdf2.inputs["Base Color"])
links.new(bsdf2.outputs["BSDF"], out2.inputs["Surface"])
bsdf2.inputs["Roughness"].default_value = 0.65
bsdf2.inputs["Specular"].default_value = 0.35

# -------------------------------
# Export GLB
# -------------------------------
os.makedirs(os.path.dirname(args.out), exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=args.out,
    export_format='GLB',
    export_texcoords=True,
    export_normals=True,
    export_image_format='AUTO',
    export_yup=True
)
print(f"✅ Exported GLB: {args.out} (bake {res}x{res})")
