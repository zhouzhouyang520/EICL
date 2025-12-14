from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class SentModel():
    def __init__(self, use_gpu, device_id):
        base_path = "./models/pre_trained_models/"
        self.model_path = base_path + "all_mpnet_base_v2"
        self.use_gpu = use_gpu
        self.device_id = device_id
        self.model, self.tokenizer = self.init_model()

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def init_model(self,):
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModel.from_pretrained(self.model_path)
        if self.use_gpu:
            model.to("cuda")
        return model, tokenizer 

    def gen_emb(self, sentences):
        # Tokenize sentences
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        #print(f"inputs: {inputs}")
        #print("--------------------------")
        
        inputs["input_ids"] = inputs["input_ids"].cuda()
        input_ids = inputs["input_ids"]
        inputs["attention_mask"] = inputs["attention_mask"].cuda()
        attention_mask = inputs["attention_mask"]
        inputs["output_attentions"] = True
        output_attentions = inputs["output_attentions"]
        #print("inputs:", inputs)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**inputs)
        
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        #print(f"Sentence embeddings: {sentence_embeddings}")
        sentence_embeddings = sentence_embeddings.detach().cpu().numpy()
        #print(f"Sentence embeddings 2: {sentence_embeddings}")
        return sentence_embeddings
