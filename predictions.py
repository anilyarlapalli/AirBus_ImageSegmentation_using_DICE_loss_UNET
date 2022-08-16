import torch
import torchvision.transforms as transforms
from skimage.io import imread
import model_unet_mini

path = r'models\model_1.pt'
imagePath = r'input\test_v2\00e90efc3.jpg'
checkpoint = torch.load(path)
net = model_unet_mini.ImageSegmentation(1)
net.load_state_dict(checkpoint['model'])
# net.load_state_dict(checkpoint['model_state_dict'])
net = net.eval()
net = net.to('cuda')
img = imread(imagePath)
img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
img = img_transform(img)
img = img.unsqueeze(0)
img = img.cuda()
pred = net(img)
k = torch.permute(pred, (0, 2, 3, 1))
k = torch.squeeze(k)
k = k.cpu().detach().numpy() 
print(k.shape)

