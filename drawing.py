"""Anatomical plates and synchronized saved-activity frames. No neural simulation."""
from __future__ import annotations
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import sparse
from anatomy import read_mesh, read_swc, view_matrix

W, H, FPS = 1920, 1080, 20
BG, INK, MUTED = "#0b151e", "#eef4f4", "#9bb0bf"
GOLD, CYAN = "#ffce80", "#63c9ce"
PLATE = (1260, 710)
FONTS = {}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def label(im, xy, value, size=28, color=INK, serif=False, center=False):
    key = (size, serif)
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype(font_manager.findfont("DejaVu Serif" if serif else "DejaVu Sans"), size)
    f = FONTS[key]
    d = ImageDraw.Draw(im)
    x, y = xy
    if center:
        x -= d.textlength(value, font=f)/2
    bb = d.textbbox((x,y),value,font=f)
    if bb[0] < 0 or bb[2] > W or bb[3] > H:
        raise ValueError(f"Off-canvas label: {value}")
    d.text((x,y),value,font=f,fill=color)


def load_geometry(ids, cache, manifest):
    R = view_matrix(18,0)
    mesh, context, neurons, hashes = [], [], {}, {}
    for rec in manifest["meshes"]:
        if rec["name"] in ("CV-anterior","CRN"):
            continue
        path = cache/rec["files"][0]
        assert sha(path) == rec["sha256"][0]
        hashes[str(path.relative_to(cache))] = sha(path)
        v,f = read_mesh(path)
        mesh.append((v@R.T)[f])
    for rec in manifest["skeletons"]:
        path = cache/rec["file"]
        assert sha(path) == rec["sha256"]
        hashes[str(path.relative_to(cache))] = sha(path)
        _, seg = read_swc(path)
        xy = (seg@R.T)[:,:,[0,2]]
        if rec["bodyId"] in set(ids.tolist()):
            neurons[rec["bodyId"]] = xy
        else:
            context.append(xy)
    assert set(neurons) == set(ids.tolist()), "Every activity column must have its exact skeleton"
    tri = np.concatenate(mesh)
    order = np.argsort(tri[:,:,1].mean(1))[::-1]
    tri = tri[order]
    lo,hi = np.percentile(tri[:,:,[0,2]].reshape(-1,2),[.05,99.95],axis=0)
    full = (lo[0]-15, hi[0]+15, lo[1]-15, hi[1]+15)
    ep = np.concatenate(list(neurons.values())).reshape(-1,2)
    center = (ep.min(0)+ep.max(0))/2
    half_y = (ep[:,1].max()-ep[:,1].min())/2+19
    half_x = half_y*PLATE[0]/PLATE[1]
    detail = (center[0]-half_x,center[0]+half_x,center[1]-half_y,center[1]+half_y)
    return tri, np.concatenate(context), [neurons[int(i)] for i in ids], full, detail, hashes


def axes(size, limits, transparent=False):
    fig = Figure(figsize=(size[0]/100,size[1]/100), dpi=100, facecolor="none" if transparent else BG)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0,0,1,1])
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(limits[:2]); ax.set_ylim(limits[2:])
    return fig, canvas, ax


def static_plate(tri, context, neurons, size, limits, context_strength=1.):
    fig,c,ax = axes(size,limits)
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
    light=np.abs(n@np.array([-.30,-.72,.62]))
    rgb=np.array([.48,.64,.73])[None]*(.5+.5*light[:,None])
    colors=np.column_stack([rgb,np.full(len(rgb),.021*context_strength)])
    ax.add_collection(PolyCollection(tri[:,:,[0,2]],facecolors=colors,edgecolors="none"))
    ax.add_collection(LineCollection(context,colors=(.44,.60,.68,min(.10*context_strength,1)),linewidths=.16))
    ax.add_collection(LineCollection(np.concatenate(neurons),colors=(.30,.49,.60,min(.17*context_strength,1)),linewidths=.25))
    c.draw()
    data=np.asarray(c.buffer_rgba())[:,:,:3].copy()
    # Save the actual Matplotlib equal-aspect transform for locator rectangles.
    points=ax.transData.transform([[limits[0],limits[2]],[limits[1],limits[3]]])
    transform={"left":float(points[0,0]),"bottom":float(size[1]-points[0,1]),
               "scale_x":float((points[1,0]-points[0,0])/(limits[1]-limits[0])),
               "scale_y":float((points[1,1]-points[0,1])/(limits[3]-limits[2]))}
    fig.clear()
    return Image.fromarray(data),transform


def activity_masks(neurons,size,limits):
    rows,cols,vals=[],[],[]
    fig,c,ax=axes(size,limits,True)
    line=LineCollection(neurons[0],colors="white",linewidths=.65)
    ax.add_collection(line)
    for j,seg in enumerate(neurons):
        line.set_segments(seg);c.draw()
        alpha=np.asarray(c.buffer_rgba())[:,:,3].copy().reshape(-1)
        idx=np.flatnonzero(alpha)
        rows.append(idx);cols.append(np.full(len(idx),j));vals.append(alpha[idx].astype(np.float32)/255)
    mat=sparse.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(size[0]*size[1],len(neurons))).tocsr()
    total=np.asarray(mat.sum(1)).ravel();active=np.flatnonzero(total)
    opacity=np.minimum(total[active]*1.4,1)[:,None]
    reduced=sparse.diags(1/total[active])@mat[active]
    assert np.allclose(np.asarray(reduced.sum(1)).ravel(),1,atol=2e-6)
    assert np.allclose(reduced@np.zeros(len(neurons)),0)
    assert np.allclose(reduced@np.ones(len(neurons)),1,atol=2e-6)
    fig.clear()
    return active,opacity,reduced


def colour(values):
    # A fixed, monotonic-lightness display map, identical across neurons/time.
    stops=np.array([[.08,.22,.34],[.14,.45,.58],[.92,.51,.23],[1.,.91,.64]])*255
    xp=np.array([0,.30,.65,1])
    return np.column_stack([np.interp(values,xp,stops[:,i]) for i in range(3)])


def paint(base,masks,rates):
    active,alpha,mat=masks
    a=np.asarray(base).copy().reshape(-1,3)
    rgb=colour(np.clip(mat@rates,0,1))
    a[active]=np.round(a[active]*(1-alpha)+rgb*alpha).astype(np.uint8)
    return Image.fromarray(a.reshape(base.height,base.width,3))


def route_panel(im,data,k,box=(1430,687,382,245)):
    d=ImageDraw.Draw(im)
    x,y,w,h=box
    # Equal metric scales in x and y; plan view from simulation ground truth.
    half=float(data["meta"]["scene"]["room_half_m"])
    scale=min(w,h)/(2*half)
    def xy(p):return (x+w/2+float(p[0])*scale,y+h/2-float(p[1])*scale)
    lo=xy([-half,half]);hi=xy([half,-half])
    d.rectangle((*lo,*hi),outline="#3d5363",width=2)
    for p in data["meta"]["scene"]["pillars"]:
        px,py=xy(p["position"]);r=float(p["radius"])*scale
        d.ellipse((px-r,py-r,px+r,py+r),fill="#334b5a")
    pts=[xy(p) for p in data["position"][:293]]
    d.line(pts,fill="#526779",width=3)
    start=(k//293)*293
    trail=[xy(p) for p in data["position"][start:k+1]]
    if len(trail)>1:d.line(trail,fill=CYAN,width=3)
    ax,ay=xy(data["position"][0]);d.ellipse((ax-7,ay-7,ax+7,ay+7),outline=INK,width=2)
    px,py=xy(data["position"][k]);a=float(data["heading"][k])
    poly=[(px+15*np.cos(a),py-15*np.sin(a)),(px+10*np.cos(a+2.5),py-10*np.sin(a+2.5)),(px+10*np.cos(a-2.5),py-10*np.sin(a-2.5))]
    d.polygon(poly,fill=GOLD)


def playback_text_free(k,data,plate,masks,locator,limits,transform):
    """Visual-only export; explanations and anatomical credit accompany the clip."""
    im=Image.new("RGB",(W,H),BG)
    anatomy=paint(plate,masks,data["rates"][k])
    im.paste(anatomy.resize((1440,811),Image.Resampling.LANCZOS),(20,134))
    im.paste(locator,(70,75))
    full,detail=limits
    x0=transform["left"]+(detail[0]-full[0])*transform["scale_x"]+70
    x1=transform["left"]+(detail[1]-full[0])*transform["scale_x"]+70
    y0=transform["bottom"]-(detail[3]-full[2])*transform["scale_y"]+75
    y1=transform["bottom"]-(detail[2]-full[2])*transform["scale_y"]+75
    d=ImageDraw.Draw(im)
    d.rectangle((x0,y0,x1,y1),outline=CYAN,width=1)
    camera=Image.fromarray(data["frames"][k,0]).convert("RGB").resize((384,384),Image.Resampling.NEAREST)
    im.paste(camera,(1496,185))
    route_panel(im,data,k,box=(1496,614,384,288))
    return im


def heading_summary(data,plate,masks):
    """Three laps by six true headings; select snapshots independently of activity."""
    im=Image.new("RGB",(W,H),BG)
    headings=list(range(0,360,60))
    active=masks[0]
    yy,xx=np.unravel_index(active,(plate.height,plate.width))
    crop=(max(0,int(xx.min())-16),max(0,int(yy.min())-16),
          min(plate.width,int(xx.max())+17),min(plate.height,int(yy.max())+17))
    records=[]
    label(im,(988,43),"Heading",30,MUTED,center=True)
    label(im,(56,107),"Lap",26,MUTED,center=True)
    for col,angle in enumerate(headings):
        label(im,(110+col*295+141,99),f"{angle}°",42,GOLD,center=True)
    for lap in range(3):
        indices=np.flatnonzero(data["segment"][:data["sample_count"]]==f"lap{lap}")
        # Use the first revolution, avoiding the second visit to 0 degrees at
        # the lap end. Circular nearest-neighbour alone mixes start/end phases.
        actual=np.rad2deg(np.unwrap(data["heading"][indices].astype(np.float64)))
        actual-=360*np.round(actual[0]/360)
        row_y=185+lap*275
        label(im,(56,row_y+95),str(lap+1),38,MUTED,center=True)
        previous=-1
        for col,angle in enumerate(headings):
            j=int(np.argmin(np.abs(actual-angle)));k=int(indices[j])
            error=float(abs(actual[j]-angle))
            assert error<2., "No close recorded heading for summary cell"
            assert k>previous, "Summary headings must follow the lap chronologically"
            previous=k
            x=110+col*295
            ImageDraw.Draw(im).rounded_rectangle((x,row_y,x+282,row_y+245),radius=10,outline="#293f4c",width=1)
            tile=ImageOps.contain(paint(plate,masks,data["rates"][k]).crop(crop),(270,225),Image.Resampling.LANCZOS)
            im.paste(tile,(x+(282-tile.width)//2,row_y+(245-tile.height)//2))
            records.append({"lap":lap+1,"column_heading_deg":angle,"sample_index":k,
                            "time_s":k/FPS,"actual_heading_deg":float(actual[j]),"heading_mismatch_deg":error})
    return im,{"rows":"Recorded laps 1, 2, 3; not training iterations",
               "columns":"Actual simulator heading, not decoded model heading",
               "selection":"Closest unwrapped heading in the first revolution of each lap; no activity-based selection or interpolation",
               "fixed_crop_pixels":crop,"cells":records}
