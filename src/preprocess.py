import pandas as pd
import numpy as np
import os
import argparse
import cv2
from scipy.interpolate import griddata

def interpolate_data(df, target_size=(224, 224)):
    """Converte coordenadas de ozônio em uma imagem 2D interpolada."""
    df['color'] = df['color'].clip(lower=0)
    points = df[['x', 'y']].values
    values = df['color'].values
    
    grid_x, grid_y = np.meshgrid(
        np.linspace(points[:,0].min(), points[:,0].max(), target_size[0]),
        np.linspace(points[:,1].min(), points[:,1].max(), target_size[1])
    )
    
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    return np.nan_to_num(grid_z, nan=0.0)

def apply_augmentation(signal):
    """Aplica técnicas de aumento de dados (ruído, desfoque, deslocamento)."""
    # Adiciona ruído branco Gaussiano
    noise_wgn = lambda s, snr: s + (np.sqrt(np.mean(s**2)) * np.power(10, -0.05 * snr)) * np.random.randn(*s.shape)
    
    return {
        "orig": signal,
        "wgn_30": noise_wgn(signal, 30),
        "blur": cv2.blur(signal, (5, 5)),
        "shift": np.roll(signal, shift=5, axis=0)
    }

def main():
    parser = argparse.ArgumentParser(description="Pré-processamento e Data Augmentation de Ozônio")
    parser.add_argument('--input', type=str, required=True, help="Pasta com arquivos .csv ou .ods brutos")
    parser.add_argument('--output', type=str, default="./data/augmented/", help="Pasta de saída para .npz")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    for file in os.listdir(args.input):
        path = os.path.join(args.input, file)
        if file.endswith('.csv'):
            df = pd.read_csv(path, skiprows=8, names=["x", "y", "color"], usecols=[0,1,2]) #
        elif file.endswith('.ods'):
            df = pd.read_excel(path, engine='odf', skiprows=8, names=["x", "y", "color"]) #
        else: continue
        
        img = interpolate_data(df)
        augs = apply_augmentation(img)
        
        base_name = os.path.splitext(file)[0]
        for name, data in augs.items():
            save_path = os.path.join(args.output, f"{name}_{base_name}.npz")
            np.savez_compressed(save_path, data=data) #
            print(f"Salvo: {save_path}")

if __name__ == "__main__":
    main()
