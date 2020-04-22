import torch
import torch.nn as nn

'''
This is the enhanced model. Note: This is NOT meant for training! Only for validating
'''

class CloudSegment(nn.Module):
    def __init__(self,
        classifier_path,
        sugar_path,
        flower_path,
        fish_path,
        gravel_path,
        classifier_class_order = ['flower', 'gravel', 'sugar', 'fish'],
        classifier_threshold = 0.5,
        dataloader_class_order = ['sugar','flower','fish','gravel'],
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):

        super(CloudSegment, self).__init__()
        self.device = device
        self.classifier = torch.load(classifier_path, map_location = device)
        self.classifier_class_order = classifier_class_order
        self.classifier_threshold = classifier_threshold
        self.dataloader_class_order = dataloader_class_order

        self.sugar_seg = torch.load(sugar_path, map_location = device)

        self.flower_seg = torch.load(flower_path, map_location = device)

        self.fish_seg = torch.load(fish_path, map_location = device)

        self.gravel_seg = torch.load(gravel_path, map_location = device)

    def forward(self, x):
        # X is an image
        class_preds_grad = self.classifier(x)
        class_preds = class_preds_grad.clone().detach()
        classifier_mask = class_preds > self.classifier_threshold
        
        # Prepare all the masks
        masks = dict(
            sugar = self.sugar_seg(x),
            flower = self.flower_seg(x),
            fish = self.fish_seg(x),
            gravel = self.gravel_seg(x)
        )

        predicted_masks = [0,0,0,0]
        indices = {clas:self.dataloader_class_order.index(clas) for clas in ['sugar','flower','fish','gravel']}
        for clas in ['sugar','flower','fish','gravel']:
            predicted_masks[indices[clas]] = masks[clas]

        # Cat the masks together
        predicted_masks = torch.cat(predicted_masks, dim = 1)

        classifier2dl_classidx = {}
        for clas in ['sugar','flower','fish','gravel']:
            classifier2dl_classidx[self.classifier_class_order.index(clas)] = self.dataloader_class_order.index(clas)
            
        final_predicted_masks = predicted_masks.clone().detach()
        for x_idx in range(class_preds.shape[0]):
            for class_idx in range(class_preds.shape[1]):
                final_predicted_masks[x_idx, classifier2dl_classidx[class_idx]] = predicted_masks[x_idx, classifier2dl_classidx[class_idx]]*class_preds[x_idx, class_idx]

        return final_predicted_masks, class_preds_grad

