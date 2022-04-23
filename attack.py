from typing import List, Iterator, Dict, Tuple, Any, Type
import torch.nn.functional as func 
import numpy as np
import torch
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        epsilon = 0.01,
        alpha = 0.005,
        steps=10,
        min_val = 0,
        max_val = 1
    ):
        """
        This file contains code for untargeted FGSM attack
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            epsilon: magnitude of perturbation that is added

        """
        self.vm = vm
        self.steps=10
        self.device = device
        self.attack_path = attack_path
        self.epsilon = 0.11
        self.alpha=self.epsilon*1/self.steps
        self.min_val = 0
        self.max_val = 1

    def attack(
        self, original_images: np.ndarray, labels: List[int], target_label = None,
    ):
        original_images = original_images.to(self.device)
        original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        original_images.requires_grad = True
        perturbed_images=self.step(original_images,target_labels)
        for i in range(self.steps-1):
            perturbed_images=self.step(perturbed_images,target_labels)
        
            perturbed_images.retain_grad()
            # get gradient with repect to labels
            data_grad = self.grad1(perturbed_images, target_labels)
            
            
            sign_data_grad = data_grad.sign()

            # perturd image
            perturbed_images = perturbed_images - self.alpha*sign_data_grad
            perturbed_images = torch.clamp(perturbed_images, self.min_val, self.max_val)

            adv_outputs = self.vm.get_batch_output(perturbed_images)
            final_pred = adv_outputs.max(1, keepdim=True)[1]
            correct = 0
            correct += (final_pred == target_labels).sum().item()
            #print("---------------------")
            #print(correct)
            #print("---------------------")
            self.alpha = 0.997 * self.alpha

            if correct == original_images.size(dim=0):
                #print("------------------")
                #print("!!!early stopping!!!")
                #print("------------------")
                break
        

        adv_outputs = self.vm.get_batch_output(perturbed_images)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        return np.squeeze(perturbed_images.cpu().detach().numpy()), correct

    def step(self,original_images, target_labels):
        
        original_images.retain_grad()
        # get gradient with repect to labels
        data_grad = self.grad1(original_images, target_labels)
        
        
        sign_data_grad = data_grad.sign()

        # perturd image
        perturbed_image = original_images - self.alpha*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val)
        return perturbed_image
    def grad1(self, batch, labels):
        self.vm.gradient_queries += batch.shape[0]
        return self.grad2(batch, labels)
    def grad2(self, original_images, labels):
        self.vm.defender.model.eval()
        outputs = self.vm.defender.model(original_images)
        loss = func.nll_loss(outputs, labels)
        self.vm.defender.model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad = original_images.grad.data
        return data_grad