# IdaLy
This repository contains code for the paper, Advancing Industrial Data Augmentation: A Thorough Survey from Foundations to Frontier Applications (submitted). IdaLy provides a python library and a software that support various algorithms for industrial data augmentation, which can promote the effiency of your industrial machine learning projects (such as fault diagnosis and soft sensor). You can use any algorithms in your script files by importing the python library. Certainly, if you are not familiar with coding of python or just want to finish augmentation tasks conveniently, you can use our software directly.  
## 1. Data Augmentation Methods
Because of the sampling difficulty and data privacy of industrial data, data we can acquire from real industry process sometimes is insufficient, thus causing problems such as overfitting and class-imblance while training intelligent agencies. Nowadays, more and more data augmentation algorithms are applied in the field of industrial big data. We collect some algorithms often used and classify them as follows:  
 ![methods](https://github.com/3uchen/IdaLy/blob/master/methods.png)  
 We collect these algorithms in a python library.You can download it in [idaly](https://github.com/3uchen/IdaLy/tree/master/src/idaly) directly. Or you can download the library by pip instrustion: `pip install idaly`. It should be noted that the library is Python-based and requires at least Python 3.6, and the envrionment should satify [requirement.txt](https://github.com/3uchen/IdaLy/blob/master/requirements.txt).  
 Here we give an example about how to use the library.  
```python
 import idaly.augmentation as aug
 
 ori_data_path = "your_ori_industrial_data_path.npy"
 out_data_path = "your_output_path.npy"
 ori_data = np.load(ori_data_path)
 
 # aug_method = aug.method(parameter)
 # data_gen = aug_method.fit(ori_data)
 aug_smote = aug.Smote(N=the numer generated, k=15)
 gen_data = aug_smote.fit(ori_data)
 
 # visualization()
 np.save(np.concentrate((ori_data, gen_data), axis=0),out_data_path)
 ```
 ## 2. Industrial Data Augmentation Platform
 ![example](https://github.com/3uchen/IdaLy/blob/master/example.png)  
 To fininsh industrial data augmentation tasks more conveniently, we intergrate algorithms mentioned above into a software developed by pyqt5. You can download the software [here](https://drive.google.com/file/d/1muqsfoieiJoRcCWeEK9OmyYlBWDwvyO4/view?usp=sharing) directly, or clone Idaly and run our python file.  
 ```
 git clone git@github.com:3uchen/IdaLy.git
 cd src\idap
 python idap_v1.py
 ```  
Idap mainly consists of seven modules: data import, algorithm select, algorithm description, parameter setting, PCA visualization, simulation test and save.
- **Data import**  
The data import module, accessed by clicking the ‘open file’ blue label in the menu bar, is mainly used for the importation of raw data and identification of industrial context. At present, the IDAS supports importing data in NumPy, Excel, and Text file formats, which includes features and labels. The IDAS will automatically recognize the industrial task type according to the label of data after importation. In a regression task scenario (e.g., KPI predictions), the IDAS regards the continuous output label as feature, while in a classification task scenario (e.g., fault diagnosis), the IDAS reorganizes the raw data based on labels and sequentially generates virtual data for each class.
- **Methods Selection and Description Module**  
As of now, the IDAS has integrated twenty IDA methods, which are categorized into three classed: transformation, interpolation and distribution estimation. Users can opt for either single algorithm mode or combined mode during the algorithm selection, after which the algorithm description module will present a brief overview of the selected along with links to references.
- **Parameter Configuration Module**  
This module allows users to preconfigure parameters, while also enabling users to dynamically adjust parameters based on results of data generation and industrial intelligence tasks assessment to achieve optimal performance.
- **PCA Visualization Module**  
After the process of data generation, PCA visualization module leverages PCA to reduce the dimensionality of the newly merged data from both the original and synthetic datasets, and subsequently plots them on the same two-dimensional scatterplot respectively, through which users can have an initial assessment of the effectiveness of data augmentation from the perspective of data diversity and fidelity of synthetic samples. It is worth mentioning that, in a classification task scenario, by clicking on the corresponding legend of a specific category on the scatterplot, the IDAS can display the visualization effect of that category.
- **Simulation test**  
IDAS embed simulation test module to further identify the effect of data augmentation. We currently provide two test scenes: soft sensor and fault diagnosis. After data augmentation procedure, Idap will input the original data and new data (original data and generated data) to train model of test scenes respectively and display test results about the trained models. We will embed more test scenes and models in the future.
- **Save**  
You can save generated data, generative model (if any) and test result by click the related buttons respectively. Or you can save all results by click the label button of "Save all".  

## 3. Benchmark (TEP dataset)
 
Based on Tennessee Eastman (TE) process dataset, We conducted a comparative experiment of 19 commonly used industrial data augmentation methods and the results are: 

<center>
  <img src="https://github.com/user-attachments/assets/11ce1907-95cd-49a2-b7b9-fbe8a0d568b0" width="100%" />
  <p>Fig. 3.1 Inference Time and Accuracy of Fault Classification under Different Data Augmentation Methods.</p>
</center>

  <img src="https://github.com/user-attachments/assets/e0a4c6ab-003c-4cd4-a225-2cacead971ee" width="60%" />
  <p>Fig. 3.2 Tiered Criteria for Different Metrics.</p>


  <img src="https://github.com/user-attachments/assets/eceb4432-53be-45bd-8b92-b82356fd97f9" width="80%" />
  <p>Fig. 3.3 Ranking of Different Data Augmentation Methods on TEP Dataset.</p>

## 4. Contact
 
  jiangxiaoyu@zju.edu.cn
  
  zhengchen0113@zju.edu.cn

 
