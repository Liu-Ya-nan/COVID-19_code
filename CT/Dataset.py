import torch
from torch.utils.data import Dataset
from torch.utils import data as torch_data
import itk
import torchvision.transforms as transforms
import os
from PIL import Image
import random


class GetLoader(Dataset):
    def __init__(self, data_root, is_pretrained=False):
        super().__init__()
        self.data_root = data_root
        self.is_pretrained = is_pretrained
        self.list_path_data = os.listdir(data_root)

        # self.Transform = transforms.Compose([
        #     transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        self.Transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        name_i = self.list_path_data[index]
        data = itk.array_from_image(itk.imread(os.path.join(self.data_root, name_i)))
        # print(data.shape)
        # data = (data - data.min()) / (data.max() - data.min())
        clip = [self.Transform(Image.fromarray(img_arr)) for img_arr in data]
        data = torch.stack(clip, 0).permute(1, 0, 2, 3)

        # label_cls_str = name_i.split('_')[-1].split('.nii')[0]
        label_cls_str = name_i.split('_')[0]

        labels = [1, 0] if label_cls_str == '1' else [0, 1]

        return torch.as_tensor(data, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.list_path_data)


class RandomRotate(object):

    def __init__(self):
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        ret_img = img.rotate(self.rotate_angle, resample=self.interpolation)

        return ret_img

    def randomize_parameters(self):
        self.rotate_angle = random.randint(-10, 10)

# # if __name__=='__main__':
#     data_root = r"/home/qwe/data/cuihaihang/lyn/CT2/256/l348nii_data/l348nii_data_test"
#     dataNII = GetLoader(data_root, is_pretrained=False)
#     # data, labels = GetLoader(data_root, is_pretrained=False)
#     # print(len(dataNII))
#     print(labels)
#
#     valid_loader = torch_data.DataLoader(dataNII, batch_size=2, shuffle=False, num_workers=4,
#                                          pin_memory=False)
# print(len(valid_loader))
# for i in range(len(dataNII)):
#     image2d, label2d = dataNII[i]
#
#     print('image size ......')
#     print(image2d.shape)  # torch.Size([3, 32, 64, 64])----torch.Size([3, 348, 512, 512])
#
#     print('label size ......')
#     print(label2d.shape)  # torch.Size([2])


#         # 可视化
#         # for j in range(image2d.shape[1]):
#         #     print(j)
#         #     oneImg = image2d[0, j, :, :]
#         #     # print(oneImg.shape)   # torch.Size([512, 512])
#         #     # plt.subplot(18, 8, j + 1)
#         #     # plt.title(j)
#         #     plt.imshow(oneImg, cmap='gray')
#         #     # plt.show()
#         #     # plt.savefig('/home/fengxiufang/cuihaihang/lyn/liuyanan/covid_3-1/CT_two/test/table '+ '{}'.format(j) + '.png')
#         #
#         #     # pylab.show()
#         #     plt.axis('off')
#         #
#         # plt.show()
