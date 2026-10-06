"""PCA and synchronized Path/PCA panels for the fixed public recording."""
import numpy as np
from PIL import Image, ImageDraw
import drawing as d

COLORS = ('#61d6cc', '#ffcb77', '#ec829c')
PANEL = '#101f2b'
LABELS = ('Path', 'PCA neurons (2D)')


def compute_pca(activity):
    """Centre model rates and use SVD; no behavioural labels or scaling by variance.

    Fit once on every recorded sample. The axes describe population activity,
    not physical coordinates. Fix signs by each loading's largest magnitude.
    """
    x = np.asarray(activity, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError('PCA requires a finite samples-by-neurons matrix')
    mean = x.mean(axis=0)
    centred = x - mean
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    loadings = vt[:2].T.copy()
    for j in range(2):
        if loadings[np.argmax(np.abs(loadings[:, j])), j] < 0:
            loadings[:, j] *= -1
    variance = singular**2
    explained = variance[:2]/variance.sum() if variance.sum() else np.zeros(2)
    return dict(scores=centred@loadings, loadings=loadings, mean=mean,
                explained_variance_ratio=explained)


def project(xy, box):
    """Fit a panel with equal scale on both axes; never align it to another panel."""
    x0, y0, x1, y1 = box
    centre = (xy.min(axis=0) + xy.max(axis=0))/2
    span = np.maximum(np.ptp(xy, axis=0), 1e-8)
    scale = min((x1-x0)/span[0], (y1-y0)/span[1])/1.18
    return np.column_stack([(x0+x1)/2+(xy[:, 0]-centre[0])*scale,
                            (y0+y1)/2-(xy[:, 1]-centre[1])*scale])


class Replay:
    """Only Path/PCA labels are drawn on the video. All panels share sample k."""

    def __init__(self, data, plate, masks, locator, limits, transform):
        self.data, self.plate, self.masks = data, plate, masks
        self.n = data['sample_count']
        self.pca = compute_pca(data['rates'][:self.n])
        self.lap = np.array([int(s.removeprefix('lap'))+1 for s in data['segment'][:self.n]])
        self.indices = [np.flatnonzero(self.lap == lap) for lap in (1, 2, 3)]
        self.base = Image.new('RGB', (d.W, d.H), d.BG)
        self.base.paste(locator, (1380, 40))
        full, detail = limits
        x0 = 1380+transform['left']+(detail[0]-full[0])*transform['scale_x']
        x1 = 1380+transform['left']+(detail[1]-full[0])*transform['scale_x']
        y0 = 40+transform['bottom']-(detail[3]-full[2])*transform['scale_y']
        y1 = 40+transform['bottom']-(detail[2]-full[2])*transform['scale_y']
        ImageDraw.Draw(self.base).rectangle((x0,y0,x1,y1), outline=d.CYAN, width=1)
        self.boxes = [(40,750,930,1040), (980,750,1880,1040)]
        coordinates = [data['position'][:self.n, :2], self.pca['scores']]
        self.points = [project(xy, (x0,y0,x1,y1-40)) for xy, (x0,y0,x1,y1) in zip(coordinates,self.boxes)]
        pen = ImageDraw.Draw(self.base)
        for title, points, box in zip(LABELS, self.points, self.boxes):
            pen.rounded_rectangle((box[0]-8,box[1]-8,box[2]+8,box[3]+8), radius=12, fill=PANEL, outline='#294150')
            for ix in self.indices:
                pen.line([tuple(p) for p in points[ix]], fill='#2b4150', width=2)
            d.label(self.base, ((box[0]+box[2])/2,box[3]-40), title, size=28, center=True)

    def frame(self, k):
        im = self.base.copy()
        anatomy = d.paint(self.plate, self.masks, self.data['rates'][k])
        im.paste(anatomy.resize((1280,721), Image.Resampling.LANCZOS), (15,10))
        camera = Image.fromarray(self.data['frames'][k,0]).convert('RGB').resize((400,400), Image.Resampling.NEAREST)
        im.paste(camera, (1400,270))
        pen = ImageDraw.Draw(im)
        for points in self.points:
            for color, indices in zip(COLORS,self.indices):
                ix = indices[indices <= k]
                if len(ix) > 1: pen.line([tuple(p) for p in points[ix]], fill=color, width=3)
            x,y = points[k]
            pen.ellipse((x-7,y-7,x+7,y+7), fill=COLORS[self.lap[k]-1], outline=d.INK, width=2)
        x,y = self.points[0][k]
        angle = float(self.data['heading'][k])
        # Draw the current position last, above every lap, with a dark halo and
        # a bright outline so it remains readable where the paths overlap.
        tip = (x+38*np.cos(angle), y-38*np.sin(angle))
        pen.line([(x,y),tip], fill=d.BG, width=9)
        pen.line([(x,y),tip], fill=d.INK, width=4)
        pen.ellipse((x-19,y-19,x+19,y+19), fill=d.BG)
        pen.ellipse((x-13,y-13,x+13,y+13), fill=COLORS[self.lap[k]-1], outline=d.INK, width=3)
        pen.ellipse((x-3,y-3,x+3,y+3), fill=d.BG)
        return im

    def figure(self):
        """Two panels with the full three-lap paths, retaining the video's colours."""
        im = Image.new('RGB',(1920,720),d.BG)
        boxes = [(40,40,930,670),(980,40,1880,670)]
        pen = ImageDraw.Draw(im)
        for title, xy, box in zip(LABELS,
                (self.data['position'][:self.n,:2],self.pca['scores']), boxes):
            pen.rounded_rectangle(box,radius=12,fill=PANEL,outline='#294150')
            pts = project(xy,(box[0]+20,box[1]+20,box[2]-20,box[3]-65))
            for color, ix in zip(COLORS,self.indices):
                pen.line([tuple(p) for p in pts[ix]],fill=color,width=3)
            d.label(im,((box[0]+box[2])/2,box[3]-45),title,size=30,center=True)
        return im
