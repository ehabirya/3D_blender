# app/deform_avatar.py — FINAL (multi-photo + FaceMask + 4K bake)
import bpy, os, sys, argparse, math
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
parser.add_argument("--frontTexList", type=str, default="")  # NEW
parser.add_argument("--sideTexList",  type=str, default="")  # NEW
parser.add_argument("--backTexList",  type=str, default="")  # NEW
parser.add_argument("--texRes", type=int, default=2048)
parser.add_argument("--highDetail", action="store_true")
parser.add_argument("--out", type=str, default="/tmp/avatar.glb")
parser.add_argument("--make_bases", action="store_true")
args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

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
    tcoord = nodes.get("TexCoord") or nodes.new("ShaderNodeTexCoord"); tcoord.name = "TexCoord"
    links.new(tcoord.outputs["Camera"], tex.inputs["Vector"])
    return tex

def _split_list(arg: str):
    return [p for p in (arg or "").split(";") if p and os.path.exists(p)]

def _load_many(nodes, links, paths):
    imgs = []
    for p in paths:
        n = _load_img_node(nodes, links, p)
        if n: imgs.append(n)
    return imgs

def _merge_images(nodes, links, tex_nodes, weights=None):
    if not tex_nodes:
        return None
    if len(tex_nodes) == 1:
        return tex_nodes[0].outputs["Color"]
    if not weights or len(weights) != len(tex_nodes):
        weights = [1.0] * len(tex_nodes)
    s = sum(weights) or 1.0
    weights = [w/s for w in weights]
    acc_color = tex_nodes[0].outputs["Color"]
    acc_w = weights[0]
    for tex, w in zip(tex_nodes[1:], weights[1:]):
        mix = nodes.new("ShaderNodeMixRGB")
        denom = nodes.new("ShaderNodeMath"); denom.operation='ADD'
        denom.inputs[0].default_value = acc_w
        denom.inputs[1].default_value = w
        div = nodes.new("ShaderNodeMath"); div.operation='DIVIDE'
        links.new(denom.outputs["Value"], div.inputs[1])
        div.inputs[0].default_value = w
        links.new(div.outputs["Value"], mix.inputs["Fac"])
        links.new(acc_color, mix.inputs[1])
        links.new(tex.outputs["Color"], mix.inputs[2])
        acc_color = mix.outputs["Color"]
        acc_w += w
    return acc_color

def _make_facemask_if_missing(obj, name="FaceMask"):
    if obj.data.attributes.get(name):
        return name
    attr = obj.data.attributes.new(name=name, type='FLOAT', domain='POINT')
    verts = obj.data.vertices
    if len(verts) == 0:
        return name
    zs = np.array([v.co.z for v in verts], dtype=np.float32)
    z_min, z_max = float(zs.min()), float(zs.max())
    z_low = z_min + 0.65 * (z_max - z_min)
    z_high = z_min + 0.95 * (z_max - z_min)
    obj.data.calc_normals()
    vals = np.zeros(len(verts), dtype=np.float32)
    for i, v in enumerate(verts):
        ny = v.normal.y
        z = v.co.z
        head_band = 0.0
        if z >= z_low:
            t = min(1.0, max(0.0, (z - z_low) / max(1e-6, (z_high - z_low))))
            head_band = t
        frontness = 1.0 if ny < -0.15 else 0.0
        vals[i] = float(head_band * frontness)
    if vals.max() > 0:
        vals = (vals / vals.max()) ** 0.7
    attr_data = obj.data.attributes[name].data
    for i, v in enumerate(attr_data):
        v.value = float(vals[i])
    return name

def _set_simple_shape(obj):
    scale = max(0.2, float(args.height) / 1.75)
    obj.scale = (scale, scale, scale)

obj = _main_mesh()
obj.name = "Avatar"
bpy.context.view_layer.objects.active = obj
_set_simple_shape(obj)
_ensure_uv(obj)

mat_name = "AvatarProjection"
mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

out = nodes.new("ShaderNodeOutputMaterial")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

frontN = _load_img_node(nodes, links, args.frontTex)
sideN  = _load_img_node(nodes, links, args.sideTex)
backN  = _load_img_node(nodes, links, args.backTex)

# NEW: multi-photo inputs -> composites
front_list = _split_list(args.frontTexList)
side_list  = _split_list(args.sideTexList)
back_list  = _split_list(args.backTexList)

front_nodes = _load_many(nodes, links, front_list) if front_list else ([frontN] if frontN else [])
side_nodes  = _load_many(nodes, links, side_list)  if side_list  else ([sideN]  if sideN  else [])
back_nodes  = _load_many(nodes, links, back_list)  if back_list  else ([backN]  if backN  else [])

front_comp = _merge_images(nodes, links, [n for n in front_nodes if n])
side_comp  = _merge_images(nodes, links, [n for n in side_nodes if n])
back_comp  = _merge_images(nodes, links, [n for n in back_nodes if n])

geom = nodes.new("ShaderNodeNewGeometry")
sep = nodes.new("ShaderNodeSeparateXYZ")
links.new(geom.outputs["Normal"], sep.inputs["Vector"])

mul_negY = nodes.new("ShaderNodeMath"); mul_negY.operation='MULTIPLY'; mul_negY.inputs[1].default_value = -1.0
links.new(sep.outputs["Y"], mul_negY.inputs[0])
front_w = nodes.new("ShaderNodeMath"); front_w.operation='MAXIMUM'; front_w.inputs[1].default_value = 0.0
links.new(mul_negY.outputs["Value"], front_w.inputs[0])

back_w = nodes.new("ShaderNodeMath"); back_w.operation='MAXIMUM'; back_w.inputs[1].default_value = 0.0
links.new(sep.outputs["Y"], back_w.inputs[0])

abs_x = nodes.new("ShaderNodeMath"); abs_x.operation='ABSOLUTE'
links.new(sep.outputs["X"], abs_x.inputs[0])
side_gain = nodes.new("ShaderNodeMath"); side_gain.operation='MULTIPLY'; side_gain.inputs[1].default_value = 0.8
links.new(abs_x.outputs["Value"], side_gain.inputs[0])

mix_fs = nodes.new("ShaderNodeMixRGB")
sum_fs = nodes.new("ShaderNodeMath"); sum_fs.operation='ADD'
links.new(front_w.outputs["Value"], sum_fs.inputs[0])
links.new(side_gain.outputs["Value"], sum_fs.inputs[1])
div_f = nodes.new("ShaderNodeMath"); div_f.operation='DIVIDE'
links.new(front_w.outputs["Value"], div_f.inputs[0])
links.new(sum_fs.outputs["Value"], div_f.inputs[1])
links.new(div_f.outputs["Value"], mix_fs.inputs["Fac"])

if front_comp and side_comp:
    links.new(front_comp, mix_fs.inputs[1])
    links.new(side_comp,  mix_fs.inputs[2])
elif front_comp:
    links.new(front_comp, mix_fs.inputs[1])
elif side_comp:
    links.new(side_comp,  mix_fs.inputs[2])

mix_all = nodes.new("ShaderNodeMixRGB")
sum_all = nodes.new("ShaderNodeMath"); sum_all.operation='ADD'
links.new(back_w.outputs["Value"], sum_all.inputs[0])
links.new(side_gain.outputs["Value"], sum_all.inputs[1])
div_b = nodes.new("ShaderNodeMath"); div_b.operation='DIVIDE'
links.new(back_w.outputs["Value"], div_b.inputs[0])
links.new(sum_all.outputs["Value"],   div_b.inputs[1])
links.new(div_b.outputs["Value"], mix_all.inputs["Fac"])

if front_comp and side_comp:
    links.new(mix_fs.outputs["Color"], mix_all.inputs[1])
elif front_comp:
    links.new(front_comp, mix_all.inputs[1])
elif side_comp:
    links.new(side_comp,  mix_all.inputs[1])

if back_comp:
    links.new(back_comp, mix_all.inputs[2])
elif backN:
    links.new(backN.outputs["Color"], mix_all.inputs[2])

fm_name = _make_facemask_if_missing(obj, "FaceMask")
attr = nodes.new("ShaderNodeAttribute"); attr.attribute_name = fm_name

face_source = front_comp if front_comp else (frontN.outputs["Color"] if frontN else None)
if face_source:
    mix_face = nodes.new("ShaderNodeMixRGB")
    links.new(attr.outputs["Fac"], mix_face.inputs["Fac"])
    links.new(face_source, mix_face.inputs[1])
    if (front_comp or side_comp or back_comp):
        links.new(mix_all.outputs["Color"], mix_face.inputs[2])
    else:
        links.new(face_source, mix_face.inputs[2])
    links.new(mix_face.outputs["Color"], bsdf.inputs["Base Color"])
else:
    if (front_comp or side_comp or back_comp):
        links.new(mix_all.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.78, 0.72, 0.68, 1.0)

bsdf.inputs["Roughness"].default_value = 0.65
bsdf.inputs["Specular"].default_value = 0.35

obj.data.materials.clear()
obj.data.materials.append(mat)

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'CPU'

res = int(args.texRes)
if args.highDetail:
    res = max(res, 4096)
else:
    res = max(1024, min(res, 4096))

bake_img = bpy.data.images.get("BakedTex") or bpy.data.images.new("BakedTex", width=res, height=res, alpha=True)
bake_node = nodes.new("ShaderNodeTexImage"); bake_node.image = bake_img
nodes.active = bake_node

_ensure_uv(obj)

bpy.context.view_layer.objects.active = obj
for o in bpy.context.selected_objects: o.select_set(False)
obj.select_set(True)

bpy.context.scene.cycles.bake_type = 'DIFFUSE'
bpy.context.scene.render.use_bake_multires = False
bpy.context.scene.render.bake.use_pass_direct = False
bpy.context.scene.render.bake.use_pass_indirect = False
bpy.context.scene.render.bake.margin = 4
bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})

nodes.clear()
out2 = nodes.new("ShaderNodeOutputMaterial")
bsdf2 = nodes.new("ShaderNodeBsdfPrincipled")
tex_baked = nodes.new("ShaderNodeTexImage"); tex_baked.image = bake_img
links = mat.node_tree.links
links.new(tex_baked.outputs["Color"], bsdf2.inputs["Base Color"])
links.new(bsdf2.outputs["BSDF"], out2.inputs["Surface"])
bsdf2.inputs["Roughness"].default_value = 0.65
bsdf2.inputs["Specular"].default_value = 0.35

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
