import torch
from monai.transforms import MapTransform
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn

class RemapLabels(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.mapping = {0: 0, 205: 1, 420: 2, 500: 3, 550: 4, 600: 5, 820: 6, 850: 7}
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.remap(d[key])
        return d

    def remap(self, tensor):
        remapped = torch.zeros_like(tensor)
        for orig, new in self.mapping.items():
            remapped[tensor == orig] = new
        # Set values that do not match the mapping to 0
        for unique_value in torch.unique(tensor):
            if unique_value.item() not in self.mapping:
                remapped[tensor == unique_value] = 0
        return remapped
    
def compute_text_embeddings(class_definitions, tokenizer, text_encoder, device):
    # Read class definitions
    class_definitions = pd.read_csv(class_definitions)
    
    # Prepare the texts by combining class names and definitions
    texts = [f"{row['class']}: {row['definition']}" for _, row in class_definitions.iterrows()]
    
    # Tokenize all texts at once
    tokenized_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Pass through text encoder to get all hidden states
    outputs = text_encoder(**tokenized_input, output_hidden_states=True)
    text_features = outputs.hidden_states  # This is a tuple of all hidden states
    
    # Collect embeddings from all layers
    txt_encoding = []
    for state in text_features:
        # Take the [CLS] token embedding (assuming it's at position 0)
        cls_embeddings = state[:, 0, :]  # Shape: [num_classes, embedding_dim]
        txt_encoding.append(cls_embeddings)
    
    # Move the embeddings to the device, as you did in your original function
    text_embeddings_all_stages = [stage_embeddings.to(device) for stage_embeddings in txt_encoding]
    
    return text_embeddings_all_stages



def CE_loss(visual_embeddings, logits_downsampled):
    """
    Computes the Cross-Entropy loss between the visual embeddings and the downsampled logits.
    Args:
        visual_embeddings (torch.Tensor): The predicted visual embeddings with shape [B, C, N], 
                                          where B is the batch size, C is the number of classes, 
                                          and N is the number of elements.
        logits_downsampled (torch.Tensor): The downsampled logits with shape [B, N], 
                                           where B is the batch size and N is the number of elements.
    Returns:
        torch.Tensor: The computed Cross-Entropy loss.
    """
    targets = torch.argmax(logits_downsampled, dim=1)  # Shape: [B, N]
    # Reshape predictions from [B, C, N] to [B * N, C]
    predictions = visual_embeddings.permute(0, 2, 1).reshape(-1, visual_embeddings.shape[1])
    targets = targets.reshape(-1)
    
    CE_loss = nn.CrossEntropyLoss(reduction="mean")
    return CE_loss(predictions, targets)

def info_nce_loss(visual_embeddings, text_embeddings, cp_feats_visual, cp_feats_text, device, temperature=0.2, noise_level = 0.01):
    # print(visual_embeddings.shape, text_embeddings[0].shape)
    # visual_embeddings = visual_embeddings[:, 1:, :]       #comment this line for MSD brain tumor dataset
    feature_embeddings = [feature_embedding[:, 0, :] for feature_embedding in text_embeddings]  # Shape: [feature_nums, num_channels-1, embedding_dim]
    
    ## GLOBAL ALIGNMENT
    weights = []
    visual_embeddings = cp_feats_visual(visual_embeddings)
    
    local_visual_embeddings = F.normalize(visual_embeddings, p=2, dim=-1)
    
    for feature_embedding in feature_embeddings:
        # Cosine similarity with visual embeddings
        feature_embedding = feature_embedding.unsqueeze(0).expand(visual_embeddings.size(0), -1, -1) # print(feature_embedding.shape, visual_embeddings.shape)  #torch.Size([1, 7, 768]) torch.Size([1, 7, 128])
        feature_embedding = cp_feats_text(feature_embedding) # print(feature_embedding.shape, visual_embeddings.shape)  #torch.Size([1, 7, 128]) torch.Size([1, 7, 128])
        feature_embedding = F.normalize(feature_embedding, p=2, dim=-1)
        similarity = F.cosine_similarity(feature_embedding, local_visual_embeddings, dim=-1)  # Shape: [num_class]
        # similarity = torch.matmul(feature_embedding, visual_embeddings.transpose(-2, -1)) 
        
        # Compute weights using softmax
        weights.append(F.softmax(similarity, dim=-1))  # Shape: [num_class]
        
    # Stack weights for all features: Shape [6, num_class]
    weights = torch.stack(weights, dim=0).squeeze(1)       
    
    # Aggregate text embeddings for each class
    aggregated_text_embeddings = torch.zeros(text_embeddings[0].shape[0], 128, device=device)  # Initialize aggregated embeddings, 128 shared dimension

    for i, feature_embedding in enumerate(feature_embeddings):
        # Weighted sum
        feature_embedding = cp_feats_text(feature_embedding)
        aggregated_text_embeddings += weights[i].unsqueeze(-1) * feature_embedding
    # Compute cosine similarity between text and visual embeddings
    aggregated_text_embeddings = aggregated_text_embeddings.unsqueeze(0).expand(visual_embeddings.size(0), -1, -1)
    
    # Normalize the embeddings
    aggregated_text_embeddings = aggregated_text_embeddings + noise_level * torch.randn_like(aggregated_text_embeddings)
    visual_embeddings = visual_embeddings + noise_level * torch.randn_like(visual_embeddings)

    normalized_visual_embeddings = F.normalize(visual_embeddings.squeeze(0), p=2, dim=-1)
    normalized_aggregated_text_embeddings = F.normalize(aggregated_text_embeddings.squeeze(0), p=2, dim=-1)

    
    # print(normalized_aggregated_text_embeddings.shape, normalized_visual_embeddings.shape)
    text_to_visual_sim = F.cosine_similarity(normalized_aggregated_text_embeddings.unsqueeze(1), normalized_visual_embeddings.unsqueeze(0), dim=-1)  
    visual_to_text_sim = F.cosine_similarity(normalized_visual_embeddings.unsqueeze(1), normalized_aggregated_text_embeddings.unsqueeze(0), dim=-1)  

    # Reshape similarity matrix to remove batch dimension for CE loss
    text_to_visual_sim = text_to_visual_sim.view(-1, text_to_visual_sim.shape[-1])  # Shape: [B * num_classes, num_classes]
    visual_to_text_sim = visual_to_text_sim.view(-1, visual_to_text_sim.shape[-1])  # Shape: [B * num_classes, num_classes]
    labels = torch.arange(text_to_visual_sim.size(0), device=text_to_visual_sim.device)

    # Compute the InfoNCE loss for both directions
    loss_t2v = F.cross_entropy(text_to_visual_sim / temperature, labels)  # Text to Visual
    loss_v2t = F.cross_entropy(visual_to_text_sim / temperature, labels)  # Visual to Text

    # Combine both losses
    loss1 = (loss_t2v + loss_v2t) / 2
    
    #LOCAL ALIGNMENT
    loss2 = 0  # Initialize local loss
    K = 3 # Number of hard negatives to select
    for i, feature_embedding in enumerate(feature_embeddings):  # Iterate over each feature embedding
        feature_embedding = feature_embedding.unsqueeze(0).expand(visual_embeddings.size(0), -1, -1) # print(feature_embedding.shape, visual_embeddings.shape)  #torch.Size([1, 7, 768]) torch.Size([1, 7, 128])
        feature_embedding = cp_feats_text(feature_embedding) # print(feature_embedding.shape, visual_embeddings.shape)  #torch.Size([1, 7, 128]) torch.Size([1, 7, 128])
        
        # Normalize both embeddings
        feature_embedding = feature_embedding + noise_level * torch.randn_like(feature_embedding)
        normalized_feature_embedding = F.normalize(feature_embedding.squeeze(0), p=2, dim=-1)
        # normalized_visual_embedding = F.normalize(visual_embeddings, p=2, dim=-1)
   
        # Compute Feature-to-Visual Similarity
        feature_to_visual_sim = F.cosine_similarity(
            normalized_feature_embedding.unsqueeze(1),
            normalized_visual_embeddings.unsqueeze(0),
            dim=-1
        )  # Shape: [num_class, num_class]

        # Compute Visual-to-Feature Similarity
        visual_to_feature_sim = F.cosine_similarity(
            normalized_visual_embeddings.unsqueeze(1),
            normalized_feature_embedding.unsqueeze(0),
            dim=-1
        )  # Shape: [num_class, num_class]
        # print(feature_to_visual_sim.shape, visual_to_feature_sim.shape)
        # **Select Hard Negatives**
        

        _, hard_negatives_f2v = torch.topk(feature_to_visual_sim, K, dim=-1, largest=False)  # Indices of top-K hard negatives
        _, hard_negatives_v2f = torch.topk(visual_to_feature_sim, K, dim=-1, largest=False)

        # Extract hard negative similarities
        hard_neg_f2v = feature_to_visual_sim.gather(dim=-1, index=hard_negatives_f2v)  # Shape: [num_class, K]
        hard_neg_v2f = visual_to_feature_sim.gather(dim=-1, index=hard_negatives_v2f)  # Shape: [num_class, K]

        
        # **Compute InfoNCE Loss with Hard Negatives**
        # loss_f2v = -torch.mean(torch.log(
        #     torch.exp(feature_to_visual_sim.diagonal() / temperature) /
        #     (torch.exp(feature_to_visual_sim.diagonal() / temperature) + torch.sum(torch.exp(hard_neg_f2v), dim=-1))
        # ))

        # loss_v2f = -torch.mean(torch.log(
        #     torch.exp(visual_to_feature_sim.diagonal() / temperature) /
        #     (torch.exp(visual_to_feature_sim.diagonal() / temperature) + torch.sum(torch.exp(hard_neg_v2f), dim=-1))
        # ))
        # Log-softmax prevents numerical issues
        similarity_matrix_f2v = F.log_softmax(feature_to_visual_sim / temperature, dim=-1)
        similarity_matrix_v2f = F.log_softmax(visual_to_feature_sim / temperature, dim=-1)

        # Compute InfoNCE Loss with stable formulation
        loss_f2v = -torch.mean(similarity_matrix_f2v.diagonal() - torch.logsumexp(hard_neg_f2v, dim=-1))
        loss_v2f = -torch.mean(similarity_matrix_v2f.diagonal() - torch.logsumexp(hard_neg_v2f, dim=-1))

        # Accumulate loss
        loss2 += (loss_f2v + loss_v2f) / 2

    # Average the loss over all features
    loss2 /= len(feature_embeddings)
    loss = loss1 + loss2
    return loss