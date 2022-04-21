from typing import List, Iterator, Dict, Tuple, Any, Type

import numpy as np
import torch
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        image_size = [32,32],
        n_population=100,
        n_generation=200,
        mask_rate=0.3,
        temperature=0.1,
        use_mask=False,
        step_size=0.1,
        child_rate=0.2,
        mutate_rate=0.2,
        l2_threshold=7.5,
        min_val = 0,
        max_val = 1
        
    ):
        """
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            min_val: minimum value of each element in original image
            max_val: maximum value of each element in original image
                     each element in perturbed image should be in the range of min_val and max_val
            attack_path: Any other sources files that you may want to use like models should be available in ./submissions/ folder and loaded by attack.py. 
                         Server doesn't load any external files. Do submit those files along with attack.py
        """
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.image_size = image_size
        self.n_population = n_population
        self.n_generation = n_generation
        self.mask_rate = mask_rate
        self.temperature = temperature
        self.use_mask = use_mask
        self.step_size = step_size
        self.child_rate = child_rate
        self.mutate_rate = mutate_rate
        self.l2_threshold = l2_threshold

    def attack(
        self, original_images: np.ndarray, labels: List[int], target_label = None,
    ):
        original_images = original_images.to(self.device)
        original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        perturbed_image = original_images
        
        # -------------------- TODO ------------------ #
        
        for i in range(len(original_images)):
            print(original_images)
            print(original_images.shape)
            print(original_images[i])
            print(original_images[i].shape)
            print(perturbed_image)
            print(perturbed_image.shape)
            print(perturbed_image[i])
            print(perturbed_image[i].shape)
            temp,abjad=self.genetic_attack(original_images[i],labels,target_labels[i])
            perturbed_image[i]=torch.from_numpy(temp).to(perturbed_image[i])
        
            

        # ------------------ END TODO ---------------- #

        adv_outputs = self.vm.get_batch_output(perturbed_image)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        return np.squeeze(perturbed_image.cpu().detach().numpy()), correct
    
    def genetic_attack(
        self, original_image: np.ndarray, labels: List[int], target_label = None,
    ):
        """
        currently this attack has 2 versions, 1 with no mask pre-defined, 1 with mask pre-defined.
        args:
            original_image: a numpy ndarray images, [1,28,28]
            labels: label of the image, a list of size 1
            target_label: target label we want the image to be classified, int
        return:
            the perturbed image
            label of that perturbed iamge
            success: whether the attack succeds
        """
        original_image = original_image.cpu().detach().numpy()
        self.original_image = np.array(original_image)
        print("target_label is: ",target_label)
        self.mask = np.random.binomial(1, self.mask_rate, size=self.image_size).astype(
            "bool"
        )
        population = self.init_population(original_image)
        examples = [(labels[0], labels[0], np.squeeze(x)) for x in population[:10]]
        success = False
        for g in range(self.n_generation):
            print("generation: ", g)
            population, output, scores, best_index = self.eval_population(
                population, target_label
            )
            if np.argmax(output[best_index, :]) == target_label:
                print(f"Attack Success!")
                success = True
                break
        
        return population[best_index], success
    
    def fitness(self, image: np.ndarray, target: int):
        """
        evaluate how fit the current image is
        return:
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
        """
        output = self._get_batch_outputs_numpy(image)
        softmax_output = np.exp(output) / np.expand_dims(
            np.sum(np.exp(output), axis=1), axis=1
        )
        scores = softmax_output[:, target]
        return output, scores
    
    
    def eval_population(self, population, target_label):
        """
        evaluate the population, pick the parents, and then crossover to get the next
        population
        args:
            population: current population, a list of images
            target_label: target label we want the imageto be classiied, int
        return:
            population: population of all the images
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
            best_indx: index of the best image in the population
        """
        output, scores = self.fitness(population, target_label)
        # --------------TODO--------------
        score_ranks = np.sort(scores)[::-1]  # Sort the scores from largeset to smallest
        #print("Ranks:")
        #print(score_ranks)
        best_index = np.argmax(scores)  # The index for the best scored candidate
        
        logits = np.exp(np.divide(scores,self.temperature))  # Exponentiate the scores after incorporating temperature
        select_probs = np.divide(logits, np.sum(logits))  # Normalize the logits between 0-1
        # ------------END TODO-------------
        if np.argmax(output[best_index, :]) == target_label:
            return population, output, scores, best_index
        # --------------TODO--------------
        # Compute the next generation of population, which is comprised of Elite, Survived, and Offspirngs
        # Elite: top scoring gene, will not be mutated
        elite = np.array([population[best_index]])
        
        # Survived: rest of the top genes that survived, mutated with some probability
        child_number=np.floor(len(score_ranks)*self.child_rate).astype(int)
        survived_number=len(score_ranks)-child_number
        
        survived = population[(np.where((scores<score_ranks[0])&(scores>=score_ranks[survived_number-1])))]  # Survived, and mutate some of them
        mutate=np.random.rand(len(survived))
        nmut=0
        for i in range(len(survived)):
            if(mutate[i]<self.mutate_rate):
                nmut=nmut+1
                survived[i]=self.perturb(survived[i])
        #print("Number of mutations: " + str(nmut))
        # Offsprings: offsprings of strong genes
        # Identify the parents of the children based on select_probs, then use crossover to produce the next generation
        children = []
        for i in range(child_number):
            parents=np.random.choice(len(population),2,replace=False,p=select_probs)
            child=np.array([self.crossover(population[parents[0]],population[parents[1]])])
            
            if(i==0):
                children=np.array(child)
            else:
                children=np.append(children,child,axis=0)
        #print("Pop before")
        #print(population.shape)
        #print("Elite")
        #print(elite.shape)
        #print("Survived from pop")
        #print(survived.shape)
        #print("New children")
        #print(children.shape)
        print("Pop after")
        
        population =np.concatenate((elite,survived,children),axis=0)
        print(population.shape)
        print("Best index")
        print(best_index)
        print("output")
        print(output)
        print("scores")
        print(scores)
        # ------------END TODO-------------
        return population, output, scores, best_index

    def perturb(self, image):
        """
        perturb a single image with some constraints and a mask
        args:
            image: the image to be perturbed
        return:
            perturbed: perturbed image
        """
        if not self.use_mask:
            adv_images = image + np.random.randn(*self.mask.shape) * self.step_size
            # perturbed = np.maximum(np.minimum(adv_images,self.original_image+0.5), self.original_image-0.5)
            delta = np.expand_dims(adv_images - self.original_image, axis=0)
            # Assume x and adv_images are batched tensors where the first dimension is
            # a batch dimension
            eps = self.l2_threshold
            mask = (
                np.linalg.norm(delta.reshape((delta.shape[0], -1)), ord=2, axis=1)
                <= eps
            )
            scaling_factor = np.linalg.norm(
                delta.reshape((delta.shape[0], -1)), ord=2, axis=1
            )
            scaling_factor[mask] = eps
            delta *= eps / scaling_factor.reshape((-1, 1, 1, 1))
            perturbed = self.original_image + delta
            perturbed = np.squeeze(np.clip(perturbed, 0, 1), axis=0)
        else:
            perturbed = np.clip(
                image + self.mask * np.random.randn(*self.mask.shape) * self.step_size,
                0,
                1,
            )
        return perturbed

    def crossover(self, x1, x2):
        """
        crossover two images to get a new one. We use a uniform distribution with p=0.5
        args:
            x1: image #1
            x2: image #2
        return:
            x_new: newly crossovered image
        """
        x_new = x1.copy()
        for i in range(x1.shape[1]):
            for j in range(x1.shape[2]):
                if np.random.uniform() < 0.5:
                    x_new[0][i][j] = x2[0][i][j]
        return x_new

    def init_population(self, original_image: np.ndarray):
        """
        Initialize the population to n_population of images. Make sure to perturbe each image.
        args:
            original_image: image to be attacked
        return:
            a list of perturbed images initialized from orignal_image
        """
        return np.array(
            [self.perturb(original_image[0]) for _ in range(self.n_population)]
        )
    
    def _get_batch_outputs_numpy(self, image: np.ndarray):
        image_tensor = torch.FloatTensor(image)
        image_tensor = image_tensor.to(self.device)

        outputs = self.vm.get_batch_output(image_tensor)

        return outputs.cpu().detach().numpy()

