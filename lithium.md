## My Project

In this final project, I applied machine learning techniques to investigate the scarcity of the existing lithium brine sources and the limitations of other extraction methods, such as hard rock and clay. Below is my report.

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

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)

