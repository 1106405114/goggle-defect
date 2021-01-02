# author: ddthuan@pdu.edu.vn

from utils.basic_lib import *

from pycocotools.coco import COCO
import pandas as pd

from config import DS_PATH

class LoadImage(torch.utils.data.Dataset):
    def __init__(self, root):
        files = os.listdir(root)
        self.img_list = list(sorted([i for i in files if i.endswith('.jpg')]))
        #self.img_list = list(sorted(listdir(root)))
                       
    def __getitem__(self, idx):
        return self.img_list[idx]
        
    def __len__(self):
        return len(self.img_list)

batch_size = 6

def get_ds_dict(dirs):
    root = "{}/{}/raw".format(DS_PATH, dirs)
    annoFile = "annotations.json"
    coco = COCO("{}/{}".format(root, annoFile))
    imgIds = coco.getImgIds()
    imgDict = coco.loadImgs(imgIds)
    imgDF = pd.DataFrame.from_dict(imgDict)
    img_pd = imgDF['file_name']
    
    dataset = LoadImage(root)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    dataset_dicts = []
    for _, img_list in enumerate(data_loader):
        for f0 in img_list:
            record = {}
            
            out_data = imgDF[imgDF['file_name']==f0]
            filename = f0
            index= int(out_data['id'])
            
            record['file_name'] = "{}/{}".format(root, filename)
            record['image_id'] = index
            record['height']= int(out_data['height'])
            record['width']= int(out_data['width'])
    
            #sampleImgIds = coco.getImgIds(imgIds = [index])
            sampleImgDict = coco.loadImgs(index)[0]
            annIds = coco.getAnnIds(imgIds=sampleImgDict['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)
            
            objs = []
            for k in range(len(anns)):
                cat_id = int(anns[k]["category_id"])
                segms = anns[k]["segmentation"]
    
                for j in range(len(segms)):
                    obj = {
                        "bbox": segms[j],
                        #"bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": cat_id,
                        "iscrowd": 0
                    }
                    objs.append(obj)
    
            record["annotations"] = objs
            
            # kiem tra de loai bo anh khong co' nhan
            if len(objs) >0:
	            dataset_dicts.append(record)
    
    return dataset_dicts

# notice this section
# DIR contains train-img and test-img MUST be format: train_(dataset name), test_(dataset name)
dict_train = get_ds_dict("train")
dict_test = get_ds_dict("test")

augs = torch.nn.Sequential(
	T.ConvertImageDtype(torch.float)
)

def collate_custom(batch):
	return tuple(zip(*batch))
	#for i, (img, target, img_id) in enumerate(data):
	#	print("img shape: ", img.shape)
	#	print("target: ", target)

class LoadData(Dataset):
	def __init__(self, dict_data):
		self.dict = dict_data
	def __getitem__(self, idx):
		item = self.dict[idx]
		img_id = item["image_id"]
		img = augs(read_image(item["file_name"]))
		boxes = []
		labels = []
		iscrowds = []
		for ins in item["annotations"]:
		    boxes.append(ins["bbox"])
		    labels.append(ins["category_id"])
		    iscrowds.append(ins["iscrowd"])
		target = {}
		target['boxes'] = torch.FloatTensor(boxes)
		target['labels'] = torch.IntTensor(labels).type(torch.int64)+1
		target['image_id'] = torch.tensor([img_id])
		#target['area'] = area
		target['iscrowd'] = torch.IntTensor(iscrowds).type(torch.int64)
		return img, target, img_id

	def __len__(self):
		return len(self.dict)


from config import BATCH_SIZE
train_dataset = LoadData(dict_train)
train_data_loader = DataLoader(train_dataset, collate_fn = collate_custom, batch_size=BATCH_SIZE)

# =============================================================================
# for i, (img, target, img_id) in enumerate(train_data_loader):
# 	print("=====================")
# 	print(target)
# =============================================================================



