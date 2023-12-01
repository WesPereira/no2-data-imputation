# Contexto e Motivação

<p align="justify">
O uso de dados de sensoriamento remoto é crucial no monitoramento das mudanças ambientais globais e na observação da Terra, especialmente na região do Pará, Brasil, onde a Amazônia enfrenta desafios devido à atividade humana. Dados do satélite Sentinel-5P e seu instrumento TROPOMI são fundamentais para monitorar a qualidade do ar e a concentração de dióxido de nitrogênio (NO2), ajudando a avaliar o impacto do desmatamento e queimadas na atmosfera amazônica.
</p>

## Objetivo

<p align="justify">
Este trabalho visa **desenvolver e avaliar modelos de Aprendizado de Máquina para estimar a concentração de NO2 na coluna troposférica**, usando dados do sensor TROPOMI do satélite Sentinel-5P. Focando na Amazônia, o estudo busca superar dificuldades na coleta de dados precisos devido à presença de nuvens, contribuindo para o monitoramento da qualidade do ar e entendimento dos impactos humanos no equilíbrio ambiental global.
</p>

# Metodologia

<p align="justify">
O trabalho foi desenvolvido seguindo uma metodologia de ciclo de experimento, conforme ilustrado na figura abaixo. Esse processo inclui desde a fase inicial de coleta de dados até as etapas subsequentes de análise dos resultados.
</p>

<p  align="center">
    <img src="assets/img/ciclo_exp.png" alt>
    <em> <b>Figura 1: </b>Ciclo de um experimento de dados</em>
</p>

<p align="justify">
No treinamento dos modelos, 50 pontos aleatórios no estado do Pará, Brasil, foram escolhidos para a seleção de variáveis de sensoriamento remoto correlacionadas ao NO2.
</p>

<p  align="center">
    <img src="assets/img/pontos_coleta.png" alt>
    <em> <b>Figura 2: </b>Pontos de coleta dos dados</em>
</p>
