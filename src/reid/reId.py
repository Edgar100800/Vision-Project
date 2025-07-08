import torch
import torchvision.transforms as T
from numpy.linalg import norm

class ReIDModel:
    def __init__(self, model_path=None, threshold=0.85):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.model = torch.nn.Sequential(
            *list(weights.children())[:-1]  # Remove the last fully connected layer
        ).to(self.device).eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.gallery = {}
        self.threshold = threshold
    
    def _extract_features(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None
        
        img_tensor = self.transform(person_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img_tensor).flatten()
        return embedding/torch.linalg.norm(embedding)
    
    def update_gallery(self, track_id, embedding):
        if track_id not in self.gallery:
            self.gallery[track_id] = []
        self.gallery[track_id].append(embedding)

    def find_match(self, new_embedding):
        best_match_id = None
        max_similarity = -1

        if not self.gallery:
            return None, 0.0
        
        for track_id, embeddings in self.gallery.items():
            gallery_embeddings = torch.stack(embeddings)
            similarities = torch.matmul(gallery_embeddings, new_embedding)
            
            current_max_sim = torch.max(similarities).item()

            if current_max_sim > max_similarity:
                max_similarity = current_max_sim
                best_match_id = track_id

        if max_similarity > self.threshold:
            return best_match_id, max_similarity
        
        return None, max_similarity