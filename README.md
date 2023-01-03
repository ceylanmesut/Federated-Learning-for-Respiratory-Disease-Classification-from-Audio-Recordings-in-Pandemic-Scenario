## Federated Learning for Respiratory Disease Classification in Hypothetical Pandemic Scenario
This repository consists of my master thesis in which we studied the capability of federated learning in hypothetical pandemic scenario over respiratory disease classification from audio signals captured from COPD, asthma, COVID-19 patients and healthy subjects.

<p align="center" width="100%">
<img src="https://github.com/ceylanmesut/Federated-Learning-for-Respiratory-Disease-Classification-from-Audio-Recordings-in-Pandemic-Scenario/blob/main/plots/plots_disease_types.png" width=90%>
</p>

### Respiratory Structure
- `data`: A directory containing our dataset.
- `models`: A directory including centralized, pretrained and federated cough classifiers.
- `notebooks`: A directory containing the Jupyter notebooks that I used for exploratory data analysis and dimensionality reduction.
- `plots`: The plots obtained during the thesis.
- `src`: A directory containing utility Python snippets.
- `study_data`: A directory containing data collection studies for respective patients.
- `classifier.py`: A Python snippet to trained centralized and pretrained cough classifier.
- `feature_extraction.py`:A Python snippet for extracting audio features from audio signals.
- `feature_extraction.py`:A Python snippet for simulating pandemic scenario in privacy-preserving manner via federated training.
- `generate_data_csv.py`:A Python snippet to create data csv file including cough paths, duration, patient ID and patient gender information.

### Pipeline
Complete pipeline of the project as follows:
1. To generate `main_data.csv`:
   ```
   python3 generate_data_csv.py
   ```
2. To conduct exploratory data analysis, outlier removal and obtain final data csv file `main_data_wo_outliers.csv`:
    ```
    jupyter notebook notebooks\eda_outlier_removal.ipynb
    ```
3. To extract audio features and generate `feature_embedding.pkl`:
    ```
    python3 feature_extraction.py
    ```
4. To reduce dimensionality of embedding and inspect visually:
    ```
    jupyter notebook notebooks\dimensionality_reduction.ipynb
    ```  
5. To obtain centralized and pretrained cough classifiers:
    ```
    python3 classifier.py
    ```  
6. To simulate federated learning in pandemic scenario:
    ```
    python3 federated_learning.py
    ```  
### Abstract
Federated Learning (FL) is a prominent machine learning paradigm that enables distributed algorithm training by exchanging gradient information between the server and edge devices without data access and storage, thus mitigating data privacy and confidentially restrictions. With the outbreak of the most recent global pandemic, COVID-19, utilization of distributed health data and collaborative efforts to develop artificial intelligence driven mechanisms gained tremendous importance in the healthcare domain to avert detrimental consequences. Despite various studies within the existing body of research in this scope, COVID-19 classification from audio recordings of the patients in a privacy-preserving fashion is undiscovered. To fill the research gap and align with these efforts, we deploy the FL scheme and formulate a hypothetical pandemic scenario in which cough is a symptom of spreading disease, mimicking the COVID-19 pandemic. We conduct a comprehensive analysis of the factors influencing this scenario and federated optimization performance such as the number of cough samples, participating patients, information exchange, and local epochs of edge devices. Based on our experiments, we propose a federated cough classifier algorithm that achieves 73% accuracy on COVID-19 and 75% overall classification performance, when 9 infected individual accumulates a total of 2535 cough samples over a 100-day period and edge devices train for 50 epochs and send their gradient information to the server model once a day. Our experiments and algorithm demonstrate the capabilities of federated learning in the hypothetical pandemic for the cough classification from audio recordings captured from smartphones task explicitly. As the first study in this scope, our work proposes a promising approach that can be further developed and employed as a pandemic detection instrument.

### Experimental Setup
Here is the experiment setup we designed in my work.

<p align="center" width="100%">
<img src="https://github.com/ceylanmesut/Federated-Learning-for-Respiratory-Disease-Classification-from-Audio-Recordings-in-Pandemic-Scenario/blob/main/plots/experimental_setup.png" width=30%>
<img src="https://github.com/ceylanmesut/Federated-Learning-for-Respiratory-Disease-Classification-from-Audio-Recordings-in-Pandemic-Scenario/blob/main/plots/data_increment.png" width=50%>
</p>


### Dataset and Feature Embedding
The dataset utilized in my work is subject to data privacy and confidentiality requirements, therefore I am not able to publicly share the dataset and corresponding feature embedding of my work.

<p align="center" width="100%">
    <img width="40%" src="https://github.com/ceylanmesut/Federated-Learning-for-Respiratory-Disease-Classification-from-Audio-Recordings-in-Pandemic-Scenario/blob/main/plots/data_distribution.JPG">
    <img width="30%" src="https://github.com/ceylanmesut/Federated-Learning-for-Respiratory-Disease-Classification-from-Audio-Recordings-in-Pandemic-Scenario/blob/main/plots/feature_embedding.png">
</p>

### Results
We demonstrate the results of the various factors impacting federated optimization paradigm.

<p align="center" width="100%">
    <img width="31.4%" src="https://github.com/ceylanmesut/Federated-Learning-for-Respiratory-Disease-Classification-from-Audio-Recordings-in-Pandemic-Scenario/blob/main/plots/r1.PNG">
    <img width="37%" src="https://github.com/ceylanmesut/Federated-Learning-for-Respiratory-Disease-Classification-from-Audio-Recordings-in-Pandemic-Scenario/blob/main/plots/r2.PNG">
    <img width="27.75%" src="https://github.com/ceylanmesut/Federated-Learning-for-Respiratory-Disease-Classification-from-Audio-Recordings-in-Pandemic-Scenario/blob/main/plots/r3.PNG">
</p>

Performance of the server model at the respective communication rounds and various data increment types is indicated below table. The best performance achieved on each disease type and respective communication round are highlighted in bold.

<p align="center" width="100%">
    <img width="95%" src="https://github.com/ceylanmesut/Federated-Learning-for-Respiratory-Disease-Classification-from-Audio-Recordings-in-Pandemic-Scenario/blob/main/plots/r4.PNG">
</p>

Below graphs display the effect of number of communication rounds to model performance during federated optimization.

<p align="center" width="100%">
    <img width="100%" src="https://github.com/ceylanmesut/Federated-Learning-for-Respiratory-Disease-Classification-from-Audio-Recordings-in-Pandemic-Scenario/blob/main/plots/acc_per_comm_rounds_diseses.png">
</p>

### License
This work is licensed under MIT License, however it is subject to author approval and permission for public and private usage.
