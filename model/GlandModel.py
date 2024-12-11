import cv2
import torch
import torchvision.transforms
from torchvision import transforms

from .LocalBinaryPattern import LocalBinaryPattern


class GlandGrading:
    def __init__(self, image, label, model):
        self.name = image
        self.path = 'ds/images/'
        self.histo = self.path + image + ".bmp"
        self.label = label
        self.roi = None
        self.roi_path = 'ds/roi/' + self.name + '.png'

        self.descriptor = None
        self.lbp = LocalBinaryPattern(8 * 3, 3)
        self.segmentation_model = model

    def get_mask(self, histo):
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ]
        )
        histo = transform(histo)
        mask = self.segmentation_model(histo.unsqueeze(0))

        mask = torchvision.transforms.functional.rgb_to_grayscale(mask)
        return mask.squeeze(0).squeeze(0)

    def extract_roi(self):
        histo = cv2.imread(self.histo, 1)
        histo = cv2.resize(histo, (256, 256))
        mask = self.get_mask(histo)
        mask = mask.cpu().detach().numpy()

        mask[mask > 0] = 255
        mask[mask <= 0] = 0

        histo = cv2.imread(self.histo, 0)
        histo = cv2.resize(histo, (256, 256))

        res = histo.copy()
        res[mask == 0] = 255
        res[mask != 0] = histo[mask != 0]
        cv2.imwrite(self.roi_path, res)
        self.roi = res

    def get_descriptors(self):
        self.descriptor = self.lbp.describe(self.roi)
