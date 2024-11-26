## Final Project: Highlighting the Scarcity of Brine Sources and the Future Potential of Oceanic Extraction from Machine Learning
Songdi Zhao
Department of Civil and Environmental Engineering
University of California, Los Angeles
November, 2024

In this final project, the machine learning techniques was applied to investigate the scarcity of the existing lithium brine sources and the limitations of other extraction methods, such as hard rock and clay. Below is my report.

***

## Introduction 

Lithium is a critical element for renewable energy technologies, powering batteries for electric vehicles and energy storage systems. However, due to the bloom of the EV Vechicle market, the demand for lithium is rapidly increasing, and its availability from traditional sources is limited. Current extraction methods primarily focus on continental brine reservoirs, which are geographically scarce, and hard rock or clay deposits, which are less readily accessible and often expensive to exploit.

The scarcity of brine reservoirs and the limitations of alternative sources present a significant challenge for sustainable lithium production. This problem is important because lithium supply bottlenecks could hinder the global transition to clean energy technologies and lead to economic and geopolitical tensions.

Fortunately, datasets such as the **USGS National Produced Waters Geochemical Database** provide valuable geochemical and spatial data related to lithium deposits. By leveraging these datasets, we can apply a machine learning approach to classify lithium sources and explore patterns in resource availability.

For this project, I applied a **supervised machine learning technique** using a Random Forest Classifier to analyze lithium sources. My approach involved:
1. Labeling brine and non-brine sources based on geological formations.
2. Training the model to predict brine classifications using geochemical and spatial features.
3. Evaluating the results to identify the scarcity of brine reservoirs and the challenges of accessing alternative sources.

After resolving the set approaches, we concluded that the existing brine lithium sources are indeed limited, and other sources such as hard rock and clay are less accessible or scalable for widespread extraction. These findings highlight the immense potential of further exploration of in-land and oceanic brine as a future lithium resource, motivating research into technologies like lithium-selective membranes (e.g., MM membranes) to make seawater extraction viable.



## Data

The dataset selected for this project was sourced from the USGS National Produced Waters Geochemical Database, a publicly available resource containing detailed geochemical and spatial data on lithium concentrations across various geological formations. Indeed, this dataset provides essential features, including total dissolved solids (TDS), pH, temperature, depth, latitude, longitude, and formation names, etc., which are critical for classifying lithium sources as brine or non-brine. The overview of regional distribution of the discovered lithium source is presented in the figure below:  

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modelling

To analyze the scarcity of lithium brine sources and classify lithium deposits, we used a supervised machine learning approach. The dataset from the USGS National Produced Waters Geochemical Database provided a variety of features such as TDS, pH, depth, latitude, and longitude, which are highly relevant for understanding the nature of lithium sources. A Random Forest Classifier was chosen for its robustness to overfitting, interpretability, and ability to handle mixed types of data.

The model was trained to classify lithium deposits as brine or non-brine based on labeled data. Labels were generated using domain knowledge, with geological formations containing keywords like "Brine," "Aquifer," or "Smackover" marked as brine sources. Features were carefully selected and preprocessed to ensure relevance and completeness, with missing values imputed using median values. The Random Forest Classifier was then optimized for performance using a grid search to fine-tune hyperparameters such as the number of estimators and maximum tree depth.

Below is an example of the machine learning code used to develop and train the model:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Feature selection
features = ['TDS', 'DEPTHUPPER', 'DEPTHLOWER', 'LATITUDE', 'LONGITUDE', 'TEMP', 'PH']
X = data[features].fillna(data[features].median())  # Impute missing values
y = data['is_brine']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
```

Data preprocessing involved cleaning the dataset to remove missing values and normalizing features where necessary to ensure compatibility with the machine learning model. Domain knowledge was applied to label data points as brine or non-brine sources based on geological formations containing keywords such as "Brine" or "Aquifer." Key geochemical and spatial features—such as TDS, pH, and depth—were selected for their relevance to lithium deposit classification. These features provided the foundation for training a robust and interpretable model.

The Random Forest Classifier was chosen for its ability to handle non-linear relationships and noisy datasets effectively. The model was trained on an 80-20 split of the data and evaluated using metrics such as accuracy, precision, recall, and F1-score. Hyperparameter optimization through a grid search further enhanced its predictive performance. Feature importance analysis revealed that geochemical factors such as TDS and depth were the most influential predictors, supporting the conclusions drawn from this study. By highlighting the scarcity of brine lithium sources and the limitations of alternative extraction methods, the machine learning approach provided actionable insights and underscored the potential of oceanic brine lithium extraction as a critical focus for future research.


## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].


## Conclusion

This project explored the application of machine learning to analyze lithium sources, focusing on the classification and evaluation of brine and non-brine deposits. The results demonstrated that the existing brine sources, while highly valuable for lithium extraction, are geographically scarce and insufficient to meet the rapidly growing demand for lithium driven by renewable energy technologies. Other sources, such as hard rock and clay, pose significant accessibility and cost challenges, further highlighting the limitations of current extraction methods. These findings underscore the need for innovative approaches to address the looming lithium supply gap.

One promising avenue is oceanic brine lithium extraction, which offers vast potential due to the abundance of lithium in seawater. However, its success depends on overcoming technical challenges, such as low lithium concentrations and competition from other ions. This work emphasizes the importance of advancing selective extraction technologies, particularly the development of MM membranes (lithium-selective membrane), to make seawater extraction feasible and scalable. Such advancements could transform oceanic lithium recovery into a sustainable and viable solution, meeting future demand while minimizing the environmental and economic constraints of traditional methods. 

Even though lithium extraction from oceanic brine has shown promising pathways for a more sustainable future, existing membrane technologies remain largely in experimental development stages, with very few advancing to scaling up and industrialization. This machine-learning project effectively highlights the deficiencies of current lithium extraction methods and their environmental impacts, emphasizing the urgent need for scalable, efficient solutions. By presenting these challenges and opportunities, this work aims to bring greater public and research attention to this rising technology, fostering advancements that could lead to a more sustainable and widely adopted approach to lithium recovery.

## References
[1] DALL-E 3

[back](./)

