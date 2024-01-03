import numpy as np
import torch
from .clip_utils import load, tokenize
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Classifier(FModule):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(1024, 512, bias=False)
        self.ln2 = nn.Linear(512, 128, bias=False)
        self.ln3 = nn.Linear(128, 101, bias=True)
    
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        return self.ln3(x)
    
class VisualPrompt(FModule):
    def __init__(self):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(16, 1, 768))    # image prompt
    def forward(self, x):
        return self.prompt
    
class TextPrompt(FModule):
    def __init__(self):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(16, 1, 512))    # image prompt
    def forward(self, x):
        return self.prompt
    
class VisualPrototype(FModule):
    def __init__(self):
        super().__init__()
        self.prototype = nn.Parameter(torch.randn(50, 1, 768))    # image prompt
    def forward(self, x):
        return self.prototype
    
class TextPrototype(FModule):
    def __init__(self):
        super().__init__()
        self.prototype = nn.Parameter(torch.randn(77, 1, 512))  # text prototype
    def forward(self, x):
        return self.prototype
        
    

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.backbone = "ViT-B/32"
        self.prompt_lengths=[16]*12 # image_layers=text_layers=12
        self.prompt_lengths[0]=0
        self.model, self.preprocess = load(self.backbone, self.prompt_lengths)
        self.image_layers=self.model.vision_layers
        self.text_layers=self.model.text_layers
        self.n_leads=2
        self.n_classes=101
        
        self.prototypes = nn.ModuleList([
            VisualPrototype(),  # image prototype
            TextPrototype()   # text prototype
        ])
        
        self.prompts = nn.ModuleList([
            VisualPrompt(),    # image prompt
            TextPrompt() # text prompt
        ])
        
        self.model, _ = load(self.backbone, self.prompt_lengths)
        self.classifier = Classifier()
        
    def full_preprocess(self, inp: list):
        # inp: [(image, text)] * B
        images, texts = zip(*inp)
        new_im = torch.stack([self.preprocess(im) for im in images], dim=0)
        new_txt = tokenize(texts, truncate=True)    # cut text to max len
        return (new_im, new_txt)
    
    @torch.no_grad()
    def query_from_prompt_pool(self, prompt_pool, query=None, attn_prtt=None):
        # prompt_pool = [(client_ids, pool)] or None
        
        n_clients, total_seq_len, dims = prompt_pool[1].size()
        k = prompt_pool[1][:, self.prompt_lengths[-1]:, ...]    # n_clients x seq_len x dim
        k = k.view(k.shape[0], -1) # n_clients x (seq_len x dim)
        dim = k.shape[1]
        
        if attn_prtt is None:
            q = query.view(1, -1)   # 1 x dim
            dim = q.shape[1]
            attn_prtt = torch.softmax((k @ q.t()) / np.sqrt(dim), dim=0)   # n_clients x 1
        new_u = (torch.sum(attn_prtt * k, dim=0)).view(total_seq_len-self.prompt_lengths[-1], 1, -1)  # seq_len x 1 x dim
        
        v = prompt_pool[1][:, :self.prompt_lengths[-1], ...]
        v = v.view(v.shape[0], -1)   # n_clients x dim
        attn_prompt = torch.softmax((k @ new_u.view(1, -1).t()) / dim, dim=0)   # n_clients x dim
        new_prompt = (torch.sum(attn_prompt * v, dim=0)).view(self.prompt_lengths[-1], 1, -1)
        
        assert new_u.shape[-1]==new_prompt.shape[-1]
        return (new_u, new_prompt, attn_prtt)
        
    @torch.no_grad()
    def adaptive_forward(self, inp: list, y: torch.Tensor, prompt_pools: list):
        image, text = inp[0]   # [(image, text)]
        miss_im = np.array(image).sum()==0
        miss_txt = len(text)==0
        v, t = self.full_preprocess(inp)
        batch_size = v.shape[0]
        assert(int(miss_im)+int(miss_txt) <=1)
        
        v_pool, t_pool = prompt_pools   # pool can be None
        if not miss_im:
            v0 = self.model.encode_image_by_steps(v, idx=[0])
            v_u, v_p, v_attn = self.query_from_prompt_pool(v_pool, v0)
            if miss_txt:
                t_u, t_p, _ = self.query_from_prompt_pool(t_pool, attn_prtt=v_attn)
            
        if not miss_txt:
            t0 = self.model.encode_text_by_steps(None, t, idx=[0])
            t_u, t_p, t_attn = self.query_from_prompt_pool(t_pool, t0)
            if miss_im:
                v_u, v_p, _ = self.query_from_prompt_pool(v_pool, attn_prtt=t_attn)
                
        self.prompts[0].prompt.data = v_p
        self.prompts[1].prompt.data = t_p
        self.prototypes[0].prototype.data = v_u
        self.prototypes[1].prototype.data = t_u
        
        v0 = torch.cat([v0 + self.prototypes[0](v0).repeat(1, batch_size, 1), self.prompts[0](v0).repeat(1, batch_size, 1)], dim=0)
        t0 = torch.cat([t0 + self.prototypes[1](t0).repeat(1, batch_size, 1), self.prompts[1](t0).repeat(1, batch_size, 1)], dim=0)
        
        image_feat = self.model.encode_image_by_steps(v0, idx=[*range(self.image_layers)][1:])
        text_feat = self.model.encode_text_by_steps(t0, text=t, idx=[*range(self.text_layers)][1:])
        
        image_feat /= image_feat.norm(dim=1, keepdim=True)
        text_feat /= text_feat.norm(dim=1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_feat @ text_feat.t()
        logits_per_text = logits_per_image.t()
        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
        
        # contrastive loss
        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        contrastive_loss = (image_loss + text_loss)*0.5
        
        fused_input = torch.cat([image_feat, text_feat], dim=1)
        output = self.classifier(fused_input)   # Bx101
        
        # task loss
        task_loss = F.cross_entropy(output, y)
        
        loss = (task_loss + 0.1*contrastive_loss) / batch_size
        return loss, output
            
    def forward(self, x, y, leads):
        '''
        x (torch.Tensor): Preprocessed image, in case missing will be zero out
        y (torch.Tensor): Tokenized text, in case missing will be zero out
        '''
        v, t = self.full_preprocess(x)
        batch_size = v.size(0)
        loss = 0.0
        
        if 0 in leads:
            v0 = self.model.encode_image_by_steps(v.to(y.device), idx=[0])
            v0 = torch.cat([v0 + self.prototypes[0](v0).repeat(1, batch_size, 1), self.prompts[0](v0).repeat(1, batch_size, 1)], dim=0)        
            image_feat = self.model.encode_image_by_steps(v0, idx=[*range(self.image_layers)][1:])
            image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
            fused_input = torch.cat([image_feat, image_feat], dim=1)
        
        if 1 in leads:     
            t0 = self.model.encode_text_by_steps(None, t.to(y.device), idx=[0])
            t0 = torch.cat([t0 + self.prototypes[1](t0).repeat(1, batch_size, 1), self.prompts[1](t0).repeat(1, batch_size, 1)], dim=0)
            text_feat = self.model.encode_text_by_steps(t0, text=t, idx=[*range(self.text_layers)][1:])
            text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
            fused_input = torch.cat([text_feat, text_feat], dim=1)
            
        if len(leads)==2:
            # cosine similarity as logits
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_feat @ text_feat.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
            
            # contrastive loss
            image_loss = F.cross_entropy(logits_per_image, labels)
            text_loss = F.cross_entropy(logits_per_text, labels)
            contrastive_loss = (image_loss + text_loss)*0.5
            loss = loss + 0.1*contrastive_loss
        
            fused_input = torch.cat([image_feat, text_feat], dim=1)
        
        output = self.classifier(fused_input)
        
        # task loss
        task_loss = F.cross_entropy(output, y)
        loss = (loss + task_loss) / batch_size
        
        return loss, output
        
class CltModel(Model):
    def __init__(self):
        super().__init__()
        
class SvrModel(Model):
    def __init__(self):
        super().__init__()
        
        
if __name__=='__main__':
    from PIL import Image
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Model().to(device)
    model.eval()
    preprocess = model.preprocess
    print(len(model.prompts[0].parameters()))
    
    img = [Image.open('./dog.jpeg') for _ in range(2)]
    txt = ["This is a photo of horse", "This is a photo of dog"]
    
    y = torch.arange(101).to(device)
    y = torch.stack([y, y], dim=0).type(torch.FloatTensor)
    
    with torch.no_grad():
        loss, output = model([img, txt], y.to(device), leads=[1])   
    print(loss.cpu().item())
    # image = [Image.fromarray(np.zeros(shape=(64, 64, 3)), mode='RGB')]
    # txt = ['']
    
    # for n, v in model.named_parameters():
    #     print(n)