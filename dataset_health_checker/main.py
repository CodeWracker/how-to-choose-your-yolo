import os
import yaml
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import argparse
import logging
from tqdm import tqdm

# Configuração de logging
logging.basicConfig(
    filename='main.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

def analisar_dataset(caminho_dataset):
    caminho_dataset = caminho_dataset.replace('\\', '/')
    logging.info(f'Iniciando análise do dataset em: {caminho_dataset}')

    # Carrega o arquivo YAML
    caminho_yaml = os.path.join(caminho_dataset, 'data.yaml').replace('\\', '/')
    try:
        with open(caminho_yaml, 'r') as arquivo_yaml:
            dados_yaml = yaml.safe_load(arquivo_yaml)
        logging.info('Arquivo data.yaml carregado com sucesso.')
    except Exception as e:
        logging.error(f'Erro ao carregar data.yaml: {e}')
        raise

    classes = dados_yaml['names']
    caminhos = {
        'train': os.path.join(caminho_dataset, 'train'),
        'val': os.path.join(caminho_dataset, 'val')
    }

    resultados = {}
    for split, caminho in caminhos.items():
        logging.info(f'Analisando split: {split}')
        caminho_images = os.path.join(caminho, 'images')
        caminho_labels = os.path.join(caminho, 'labels')

        total_imagens = 0
        total_anotacoes = 0
        imagens_sem_anotacao = 0
        anotacoes_vazias = 0
        classes_contagem = Counter()
        heatmap = np.zeros((1000, 1000))

        arquivos_imagens = [
            f for f in os.listdir(caminho_images) if f.endswith(('.jpg', '.png', '.jpeg'))
        ]

        for nome_arquivo in tqdm(arquivos_imagens, desc=f'Processando {split}', unit='imagem'):
            total_imagens += 1
            logging.info(f'Processando imagem: {nome_arquivo} e pegando a label {os.path.splitext(nome_arquivo)[0]}.txt')
            caminho_label = os.path.join(caminho_labels, os.path.splitext(nome_arquivo)[0] + '.txt')

            if not os.path.exists(caminho_label):
                imagens_sem_anotacao += 1
                logging.warning(f'Anotação ausente para a imagem: {nome_arquivo}')
                continue

            try:
                with open(caminho_label, 'r') as arquivo:
                    linhas = arquivo.readlines()
                    if not linhas:
                        anotacoes_vazias += 1
                        logging.info(f'Anotação nula para a imagem: {nome_arquivo}')
                        continue

                    for linha in linhas:
                        total_anotacoes += 1
                        dados = linha.strip().split()
                        classe_id = int(dados[0])
                        classes_contagem[classe_id] += 1

                        # Coordenadas normalizadas
                        x_center = float(dados[1])
                        y_center = float(dados[2])
                        box_largura = float(dados[3])
                        box_altura = float(dados[4])

                        # Heatmap
                        x_min = int((x_center - box_largura / 2) * 1000)
                        x_max = int((x_center + box_largura / 2) * 1000)
                        y_min = int((y_center - box_altura / 2) * 1000)
                        y_max = int((y_center + box_altura / 2) * 1000)

                        heatmap[y_min:y_max, x_min:x_max] += 1
            except Exception as e:
                logging.warning(f'Erro ao processar {nome_arquivo}: {e}')

        # Métricas finais
        resultados[split] = {
            'total_imagens': total_imagens,
            'total_anotacoes': total_anotacoes,
            'imagens_sem_anotacao': imagens_sem_anotacao,
            'anotacoes_vazias': anotacoes_vazias,
            'classes_contagem': dict(classes_contagem),
        }

        logging.info(f'{split} - Total de imagens: {total_imagens}')
        logging.info(f'{split} - Total de anotações: {total_anotacoes}')
        logging.info(f'{split} - Imagens sem anotação: {imagens_sem_anotacao}')
        logging.info(f'{split} - Anotações nulas: {anotacoes_vazias}')

        pasta_health = os.path.join(caminho_dataset, 'health')
        os.makedirs(pasta_health, exist_ok=True)

        # Salvar distribuição de classes
        plt.figure()
        plt.bar(classes_contagem.keys(), classes_contagem.values())
        plt.title(f'Distribuição de Classes - {split}')
        plt.xlabel('ID da Classe')
        plt.ylabel('Quantidade')
        plt.savefig(os.path.join(pasta_health, f'class_distribution_{split}.png'))
        plt.close()
        logging.info(f'Distribuição de classes salva para {split}.')

        # Salvar heatmap
        plt.figure()
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.title(f'Heatmap de Anotações - {split}')
        plt.savefig(os.path.join(pasta_health, f'heatmap_{split}.png'))
        plt.close()
        logging.info(f'Heatmap salvo para {split}.')

    # Salvar resultados gerais
    resultados_yaml = os.path.join(pasta_health, 'resultados.yml')
    with open(resultados_yaml, 'w') as arquivo_resultados:
        yaml.dump(resultados, arquivo_resultados)

    logging.info('Análise concluída com sucesso.')
    print('✅ Análise concluída. Resultados salvos na pasta health.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Análise de Dataset YOLO')
    parser.add_argument('dataset_path', type=str, help='Caminho para o dataset YOLO')
    args = parser.parse_args()

    try:
        analisar_dataset(args.dataset_path)
    except Exception as e:
        logging.critical(f'Erro fatal durante a análise: {e}')
        print(f'❌ Erro fatal: {e}')
