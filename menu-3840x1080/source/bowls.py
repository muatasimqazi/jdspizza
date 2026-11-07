import sys, numpy as np, glob
from rembg import new_session, remove
from PIL import Image
s = new_session('sam')
SRC = '/mnt/user-data/uploads/jdspizza/images/'
files = sys.argv[1:]
for f in files:
    im = Image.open(SRC + f).convert('RGB')
    out = remove(im, session=s, sam_prompt=[{'type': 'rectangle', 'data': [790, 0, 1300, 300], 'label': 1}], only_mask=True)
    out.save('bowl/' + f.rsplit('.', 1)[0].strip() + '.png')
    a = np.array(out) > 128
    ys, xs = np.nonzero(a)
    print(f, a.sum(), (xs.min(), xs.max(), ys.min(), ys.max()) if a.any() else None, flush=True)
