from basic_lib import *
torch.manual_seed(0)

#print(torch.__version__)
#print(torchvision.__version__)
#print(torch.version.cuda)

#dir_path = {sys.argv[1], sys.argv[2], sys.argv[3]}
feature_name = ['feat1', 'feat2', 'feat3']

augs = nn.Sequential(
	T.ConvertImageDtype(torch.float)
)

# doc 1 folder anh
# tra ve danh sach ten anh va tensor
def read_img_dir(dir_path):
	img_names = [i for i in os.listdir(dir_path)]
	img_ts_list = [augs(read_image("{}/{}".format(dir_path, i))) for i in img_names]
	img_ts = torch.stack(img_ts_list, dim=0)
	return img_names, img_ts

features = [read_img_dir(i) for i in feature_name]
final_feat = {feature_name[i]: img_ts for i, (img_names, img_ts) in enumerate(features)}

print("feat1: ", final_feat['feat1'].shape)
print("feat2: ", final_feat['feat2'].shape)
print("feat3: ", final_feat['feat3'].shape)

m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat2'],  50, 2)

image_sizes = [(512, 512)]

boxes = torch.rand(6, 4) * 64; boxes[:, 2:] += boxes[:, :2]
print(boxes)


out = m(final_feat, [boxes], image_sizes)
print(out.shape)
save_image(out, "out.jpg")



