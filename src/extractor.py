import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import numpy as np
import pandas as pd
import os
import argparse

def get_extractor():
    """Carrega a ResNet18 pré-treinada e remove a última camada."""
    resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
    extractor.eval()
    return extractor

def main():
    parser = argparse.ArgumentParser(description="Extração de features com ResNet18")
    parser.add_argument('--input', type=str, required=True, help="Pasta com arquivos .npz")
    parser.add_argument('--output', type=str, default="features.csv")
    args = parser.parse_args()

    model = get_extractor()
    features, labels, names = [], [], []
    
    # Mapeamento de rótulos baseado nos códigos originais
    class_map = {f'{i}{j}': i*4+j for i in range(3) for j in range(4)}

    for f in os.listdir(args.input):
        if not f.endswith('.npz'): continue
        
        # Carrega e prepara o tensor para a ResNet
        data = np.load(os.path.join(args.input, f))['data'].astype(np.float32)
        tensor = torch.from_numpy(data).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        
        with torch.no_grad():
            feat = model(tensor).flatten().numpy()
        
        # Extração automática do rótulo pelo nome do arquivo
        label_code = f.split('_')[-1].replace('.npz', '')
        
        features.append(feat)
        labels.append(class_map.get(label_code, -1))
        names.append(f)

    df = pd.DataFrame(features)
    df['file_name'], df['label'] = names, labels
    df.to_csv(args.output, index=False)
    print(f"Sucesso! {len(features)} amostras extraídas em: {args.output}")

if __name__ == "__main__":
    main()
