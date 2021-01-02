from utils.basic_lib import *

img_dir = sys.argv[1]

class getImg(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        files = os.listdir(img_dir)
        self.imgs = list(sorted([img for img in files]))
        
    def __getitem__(self, idx):
        img_ts = to_tensor(Image.open("{}/{}".format(self.img_dir,self.imgs[idx])))
        return self.imgs[idx], img_ts
    def __len__(self):
        return len(self.imgs)
    
ds = getImg(img_dir)
loader = DataLoader(ds, batch_size=1)
for i, (img_name, img_ts) in enumerate(loader):
    print(img_ts[0].shape)