import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Localizando os diretórios de modelo na estrutura de arquivos
model_dirs = sorted(glob.glob('plots/final/*'))

# Lendo os dados dos arquivos CSV
data = {}
for model_dir in model_dirs:
    model_name = os.path.basename(model_dir)
    file_path = os.path.join(model_dir, 'infers.csv')
    if os.path.exists(file_path):
        data[model_name] = pd.read_csv(file_path)

# Calculando os erros absolutos (diferença absoluta entre valores reais e preditos) para cada modelo
errors = {model: abs(df['yt'] - df['yp']) for model, df in data.items()}

# Definindo cores opacas distintas para cada boxplot
colors = ['red', 'green', 'blue', 'purple', 'orange', 'pink', 'brown', 'gray']

# Criando o gráfico de boxplot
# Criando o gráfico de boxplot com boxplots mais largos
plt.figure(figsize=(12, 6))
for i, (model, error) in enumerate(errors.items()):
    plt.boxplot(error, positions=[i], widths=0.6, patch_artist=True,  # Aumentando a largura dos boxplots
                boxprops=dict(facecolor=colors[i % len(colors)], color='black'),
                medianprops=dict(color='black'))

# Configurações finais do gráfico
plt.xticks(range(len(errors)), errors.keys())
plt.title('Distribuição de erros absolutos por modelo')
plt.ylabel('Erro Absoluto (|yt - yp|)')
plt.xlabel('Modelo')
plt.grid(axis='y')

# Salvando o plot em formato PDF com bbox_inches='tight'
plot_file_path = 'models_error_dist.png'
plt.savefig(plot_file_path, bbox_inches='tight')

plt.show()

