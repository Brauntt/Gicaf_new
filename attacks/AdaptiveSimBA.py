from typing import Optional, Type
from attacks.SparseSimBA import SparseSimBA
from interface.ModelBase import ModelBase
from interface.LoggerBase import LoggerBase
from numpy import clip, argwhere, zeros, array, ndarray
from sys import setrecursionlimit
from numpy.linalg import norm
from numpy.random import randint, uniform

class AdaptiveSimBA(SparseSimBA):

    def __init__(
        self, 
        size: int = 1, 
        epsilon: int = 64, 
        epsilon_multiplier: float = 2.0,
        decay_rate: float = 0.25,
        decay_period: int = 200,
        probability_rate: float = 0.01
    ) -> None: 
        self.size = size
        self.initial_epsilon = epsilon
        self.epsilon_multiplier = epsilon_multiplier
        self.decay_rate = decay_rate
        self.decay_period = decay_period
        self.probability_rate = probability_rate

    def __call__(self, 
        image: ndarray, 
        model: Type[ModelBase], 
        logger: Type[LoggerBase], 
        ground_truth: Optional[int] = None,
        target: Optional[int] = None,
        query_limit: int = 5000
    ) -> Optional[ndarray]: 
        if target:
            raise NotImplementedError("Targeted Adaptive SimBA has not been implemented yet") 
        if ground_truth == None:
            raise ValueError('Adaptive SimBA is not intended for generating false positives, please provide a ground truth')
        self.model = model
        self.height = self.model.metadata['height']
        self.width = self.model.metadata['width']
        self.channels = self.model.metadata['channels']
        self.bounds = self.model.metadata['bounds']
        self.logger = logger
        self.query_limit = query_limit
        loss_label = ground_truth
        self.epsilon = self.initial_epsilon*self.epsilon_multiplier
        self.last_epsilon_change = 0

        setrecursionlimit(max(1000, int(self.height*self.width*self.channels/self.size/self.size*10))) #for deep recursion diretion sampling

        top_preds = self.model.get_top_5(image)
        noisy_top_5_preds = self.adjust_preds(top_preds)
        top_1_label, _ = top_preds[0]

        self.logger.nl(['iterations', 'epsilon','size', 
                        'is_adv', 'image', 'top_preds', 'success', 'p'])

        self.ps = [noisy_top_5_preds[0][1]]
        self.count = 0
        past_qs = []
        self.done = []
        delta = 0
        is_adv = self.is_adversarial(top_1_label, loss_label)
        iteration = 0
        done = []
        
        # log step 0
        adv = clip(image + delta, self.bounds[0], self.bounds[1])

        self.logger.append({
            "iterations": iteration,
            "epsilon": self.epsilon,
            "size": self.size,
            "is_adv": is_adv,
            "image": image,
            "top_preds": top_preds,
            "success": False,
            "p": self.ps[-1],
        }, image, adv)

        while ((not is_adv) & (self.model.get_query_count() <= self.query_limit)): 
            iteration += 1    

            q, done = self.new_q_direction(done)

            delta, top_preds, success = self.check_pos(image, delta, q, loss_label)
            if success:
                q = -q
            self.count += self.probability_rate*100
            if not success:
                delta, top_preds, success = self.check_neg(image, delta, q, loss_label)
                self.count += self.probability_rate*100

            adv = clip(image + delta, self.bounds[0], self.bounds[1])
            
            if (self.model.get_query_count() % self.decay_period < 2 
                and self.epsilon != self.initial_epsilon 
                and self.model.get_query_count() - self.last_epsilon_change > self.decay_period - 2):
                self.epsilon = self.epsilon - self.initial_epsilon*(self.epsilon_multiplier - 1)/(1/self.decay_rate)
                self.last_epsilon_change = self.model.get_query_count()

            if success:
                self.count = 0
                past_qs.append(q)
            else:
                if uniform(0, 100, 1) < self.count:
                    if len(past_qs) > 0:
                        last_q, past_qs = past_qs[-1], past_qs[:-1]
                        delta = delta + self.epsilon * last_q
                        self.ps = self.ps[:-1]
                        self.count = 0

            if iteration % 100 == 0: # only save image and probs every 100 steps, to save memory space
                image_save = adv
                preds_save = top_preds
            else:
                image_save = None
                preds_save = None
                
            self.logger.append({
                "iterations": iteration,
                "epsilon": self.epsilon,
                "size": self.size,
                "is_adv": is_adv,
                "image": image_save,
                "top_preds": preds_save,
                "success": success,
                "p": self.ps[-1],
            }, image, adv)

            # check if image is now adversarial
            if ((not is_adv) and (self.is_adversarial(top_preds[0][0], loss_label))):
                is_adv = 1
                self.logger.append({
                    "iterations": iteration,
                    "epsilon": self.epsilon,
                    "size": self.size,
                    "is_adv": is_adv,
                    "image": adv,
                    "top_preds": top_preds,
                    "success": success,
                    "p": self.ps[-1],
                }, image, adv) 
                return adv
                
        return None

    def check_pos(self, x, delta, q, loss_label):
        success = False 
        pos_x = x + delta + self.epsilon * q
        pos_x = clip(pos_x, self.bounds[0], self.bounds[1])
        top_5_preds = self.model.get_top_5(pos_x)
        if self.model.metadata['activation bits'] <= 8:
            noisy_top_5_preds = self.adjust_preds(top_5_preds)

        idx = argwhere(loss_label==top_5_preds[:,0]) # positions of occurences of label in preds
        if len(idx) == 0:
            print("{} does not appear in top_preds".format(loss_label))
            delta = delta - self.epsilon*q # add new perturbation to total perturbation
            success = True
            return delta, top_5_preds, success
        idx = idx[0][0]
        if self.model.metadata['activation bits'] <= 8:
            p_test = noisy_top_5_preds[idx][1]
        else:
            p_test = top_5_preds[idx][1]
        if p_test < self.ps[-1] or idx != 0:
            delta = delta + self.epsilon*q # add new perturbation to total perturbation
            self.ps.append(p_test) # update new p
            success = True
        return delta, top_5_preds, success

    def check_neg(self, x, delta, q, loss_label):
        success = False
        neg_x = x + delta - self.epsilon * q
        neg_x = clip(neg_x, self.bounds[0], self.bounds[1])
        top_5_preds = self.model.get_top_5(neg_x)
        if self.model.metadata['activation bits'] <= 8:
            noisy_top_5_preds = self.adjust_preds(top_5_preds)

        idx = argwhere(loss_label==top_5_preds[:,0]) # positions of occurences of label in preds
        if len(idx) == 0:
            print("{} does not appear in top_preds".format(loss_label))
            delta = delta - self.epsilon*q # add new perturbation to total perturbation
            success = True
            return delta, top_5_preds, success
        idx = idx[0][0]
        if self.model.metadata['activation bits'] <= 8:
            p_test = noisy_top_5_preds[idx][1]
        else:
            p_test = top_5_preds[idx][1]
        if p_test < self.ps[-1] or idx != 0:
            delta = delta - self.epsilon*q # add new perturbation to total perturbation
            self.ps.append(p_test) # update new p 
            success = True
        return delta, top_5_preds, success

    def adjust_preds(self, preds):
        probs = list(map(lambda x: x[1] + uniform(low=-0.000005, high=0.000005, size=1)[0], preds))
        preds = list(map(lambda x: x[0], preds))
        return array(list(map(lambda x: array(x), zip(preds, probs))))
